import logging
import os
import time
from collections import defaultdict
from random import shuffle, sample

import torch
from torch.autograd import Variable
import numpy as np
from DISAE.trainer import Trainer, load_data
import torch.nn as nn
import wandb
md_logger = logging.getLogger(__name__)


def load_ul_data(edges_dict):
    """ Load unlabeled edges

    Args:
        edges_dict (dict): dictionary of compound-protein pairs.

    Return:
        edges (list): list of edges represented by tuples as (compound ID, protein ID).
    """
    count = 0
    count_skipped = 0
    # edges: list of tuples (chemical_ID,protein_ID)
    edges = []
    chems = []
    prots = []
    for cp in edges_dict.keys():
        chem, prot = cp.strip().split("\t")
        chems.append(chem)
        prots.append(prot)
        count += 1
        edges.append((chem, prot))
    chems = list(set(chems))
    prots = list(set(prots))
    md_logger.info(
        f"{len(chems)} chemicals, {len(prots)} proteins, loaded for unlabeled data."
        f"{count_skipped} chemicals skipped for non-Mol conversion."
    )
    return edges


class MPLTrainer(Trainer):
    def __init__(
        self,
        teacher,
        student,
        tokenizer,
        chemid2smiles,
        chemid2mol,
        prot2triplets,
        prediction_mode,
        protein_embedding_type,
        epoch=1,
        batch_size=32,
        ckpt_dir=None,
        optimizer="adam",
        l2=1e-4,
        teacher_lr=None,
        student_lr=None,
        scheduler="cosineannealing",
        freezing_epochs=0,
        temperature=0.7
    ):
        """ Class for Meta Pseudo Label training.

        Args:
            teacher: the teacher model.
            student: the student model.
            tokenizer (BertTokenizer): the tokenizer instance for tokenizing protein
                sequences.
            chemid2smiles (dict): map from chemical compound IDs to SMILES.
            chemid2mol (dict): map from chemical compound IDs to molecular graphs.
            prot2triplets (dict): map from protein IDs to triplets.
            prediction_mode (str): "binary" or "continuous".
            protein_embedding_type (str): "LSTM" or "ALBERT".
            epoch (int): number of training epochs.
            batch_size (int): batch size.
            ckpt_dir (str): checkpoint directory path. Path to save training results.
            optimizer (str): name of the optimizer. "adam" or "sgd". Default is "adam".
            l2 (float): L2 regularization rate. Default is 1e-4.
            teacher_lr (float): learning rate for the teacher model.
            student_lr (float): learning rate for the student model.
            scheduler (str): name of the scheduler. "cosineannealing" or "cyclic".
                Default is "cosineannealing".
            freezing_epochs (int): number of epochs to freeze the teacher model at the
                beginning of trainng.
        """

        self.teacher = teacher
        self.student = student
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.train_epoch = epoch
        self.optimizer = optimizer
        self.l2 = l2
        self.teacher_lr = teacher_lr
        self.student_lr = student_lr
        self.freezing_epochs = freezing_epochs
        self.prediction_mode = prediction_mode
        self.prottype = protein_embedding_type
        self.uniprot2triplets = prot2triplets
        self.chemid2smiles = chemid2smiles
        self.chemid2mol = chemid2mol
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.temperature = temperature
        if self.scheduler.lower() == "cyclic":
            self.optimizer = "sgd"
            md_logger.info(
                "CyclicLR scheduler is used. Optimizer is set to {}".format(
                    self.optimizer.upper()
                )
            )

    def _init_optimizer(self, parameters, lr):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=self.l2)
            md_logger.info(
                "Optimizer {}, LR {}, Weight Decay {}".format(
                    self.optimizer, lr, self.l2
                )
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=self.l2)
            md_logger.info(
                "Optimizer {}, LR {}, Weight Decay {}".format(
                    self.optimizer, lr, self.l2
                )
            )
        return optimizer

    def _prepare_batch(
        self,
        batch_,
        train_labels,
        train_pairs,
        unlabeled_idxs,
        unlabeled_pairs,
        batch_prep_time,
    ):
        batch_size = self.batch_size
        stime = time.time()
        ul_choices = unlabeled_idxs[batch_ * batch_size : (batch_ + 1) * batch_size]
        batch_ul_pairs = [unlabeled_pairs[idx] for idx in ul_choices]
        train_data_idxs = list(range(len(train_labels)))
        choices = sample(train_data_idxs, batch_size)
        labels = torch.tensor([train_labels[idx] for idx in choices]).cuda()
        # ----------------------------------
        #           process input
        # ----------------------------------
        batch_train_pairs = [train_pairs[idx] for idx in choices]
        batch_prot_repr = self.get_prot_repr_from_pairs(batch_train_pairs)
        batch_chem_embed = self._prepare_chem_batch_for_gin(batch_train_pairs)
        batch_chem_embed = batch_chem_embed.to("cuda:0")
        batch_ul_prot_repr = self.get_prot_repr_from_pairs(batch_ul_pairs)
        batch_ul_chem_embed = self._prepare_chem_batch_for_gin(batch_ul_pairs)
        batch_ul_chem_embed = batch_ul_chem_embed.to("cuda:0")
        # ----- move variables to GPU
        if isinstance(batch_prot_repr, Variable) and torch.cuda.is_available():
            batch_prot_repr = batch_prot_repr.cuda()
            batch_ul_prot_repr = batch_ul_prot_repr.cuda()
        batch = {
            "protein": batch_prot_repr,
            "ligand": batch_chem_embed,
        }
        ul_batch = {
            "protein": batch_ul_prot_repr,
            "ligand": batch_ul_chem_embed,
        }
        batch_prep_t = time.time() - stime
        batch_prep_time += batch_prep_t
        return ul_batch, batch, labels

    def _predict_batch(self, model, batch, activation_fn=None):
        model.eval()
        predictions = model(batch)
        if activation_fn is None:
            return predictions
        else:
            return activation_fn(predictions)

    def _update_student(
        self,
        pseudo_labels,
        ul_batch,
        optimizer,
        scheduler,
        loss_fn=torch.nn.CrossEntropyLoss(),
    ):
        self.student.train()
        batch_logits = self.student(ul_batch)
        md_logger.debug(batch_logits)
        # labels = torch.argmax(pseudo_labels, dim=1).long()
        # md_logger.debug(labels)
        # loss = loss_fn(batch_logits, labels)
        #pseudo_labels =  pseudo_labels.long()
        pseudo_labels = torch.softmax(pseudo_labels.detach() / self.temperature, dim=-1) # apply temperature on the soft lbs
        loss =nn.BCEWithLogitsLoss()(batch_logits, pseudo_labels) # treat it as a multi class task
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss

    def _update_teacher(
        self,
        labels,
        #pseudo_labels,
        student_predict,
        optimizer,
        scheduler,
        loss_fn=torch.nn.CrossEntropyLoss(),
    ):
        self.teacher.train()
        labels = labels.long()
        # predictions = torch.argmax(student_predict, dim=1)
        # if add t_sup_loss (add pseudo_labels input as well)
        #loss = loss_fn(student_predict, labels) + t_s_loss
        loss = loss_fn(student_predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss

    def train(
        self,
        edges,
        unlabeled_edges,
        train_evaluator,
        dev_edges,
        dev_evaluator,
        test_edges,
        test_evaluator,
        checkpoint_dir,
        blc_up,
        blc_low,
        save_model=False
    ):
        """ The training method.

        Args:
            edges (dict): dictionary contains labeled compound-protein pairs for
                training.
            unalbeled_edges (dict): dictionary contains unlabeled compound-protein
                pairs.
            train_evaluator (Evaluator): evaluator instance to evaluate training steps.
            dev_edges (dict): dictionary contains labeled compound-protein pairs for
                evaluation step.
            dev_evaluator (Evaluator): evaluator instance for evaluation steps.
            test_edges (dict): dictionary contains labeled compound-protein pairs for
                testing step.
            test_evaluator (Evaluator): evaluator instance for testing steps.
            checkpoint_dir (str): path to save training results.
        """
        # ----------------------------------
        #    set up data/parameters/models
        # ----------------------------------
        t_parameters = list(self.teacher.parameters())
        s_parameters = list(self.student.parameters())
        t_optimizer = self._init_optimizer(t_parameters + s_parameters, self.teacher_lr)
        s_optimizer = self._init_optimizer(s_parameters, self.student_lr)
        t_scheduler = self._init_scheduler(t_optimizer)
        s_scheduler = self._init_scheduler(s_optimizer)
        # ..............
        unlabeled_pairs = load_ul_data(unlabeled_edges)
        train_pairs, train_labels = load_data(edges, datatype="train")
        dev_pairs, dev_labels = load_data(dev_edges, datatype="dev")
        test_pairs, test_labels = load_data(test_edges, datatype="test")
        # ..............
        best_target_metric = -np.inf
        best_epoch = 0
        batch_size = self.batch_size
        batch_per_epoch = int(np.floor(len(unlabeled_pairs) / batch_size))
        record_dict = defaultdict(list)
        print("Epoch\tData\tAcc\tF1\tAUC\tAUPR")

        t_losses, s_losses = [], []
        acc_train, f1_train, auc_train, aupr_train = [], [], [], []
        acc_dev, f1_dev, auc_dev, aupr_dev = [], [], [], []
        acc_test, f1_test, auc_test, aupr_test = [], [], [], []

        # ----------------------------------
        #           training
        # ----------------------------------
        #for epoch in range(1, self.train_epoch + 1):
        epoch = 0
        while epoch < self.train_epoch + 1:
            try:
                good_epoch = False
                batch_prep_time = 0
                fail_cnt=0
                unlabeled_idxs = list(range(len(unlabeled_pairs)))
                shuffle(unlabeled_idxs)

                for batch_idx in range(batch_per_epoch):

                    if (batch_idx + 1) % 500 == 0:
                        md_logger.info(
                            f"Epoch: {epoch}, Batch: {batch_idx+1}/{batch_per_epoch}"
                        )
                    # the try except loop is used to control the balance of pseudo labels

                    ul_batch, batch, labels = self._prepare_batch(
                        batch_idx,
                        train_labels,
                        train_pairs,
                        unlabeled_idxs,
                        unlabeled_pairs,
                        batch_prep_time,
                    )
                    pseudo_labels = self._predict_batch(self.teacher, ul_batch)
                    hard_pseudo_labels = torch.argmax(pseudo_labels, dim=1).long()
                    ratio = (hard_pseudo_labels==1).sum()/(hard_pseudo_labels==0).sum()
                    if ratio > blc_low and ratio <blc_up:
                        good_epoch=True
                        s_loss = self._update_student(
                            pseudo_labels, ul_batch, s_optimizer, s_scheduler
                        )
                        s_losses.append(s_loss.item())
                        if epoch > self.freezing_epochs:
                            student_predict = self._predict_batch(self.student, batch)
                            t_loss = self._update_teacher(
                                labels, student_predict, t_optimizer, t_scheduler
                            )
                            t_losses.append(t_loss.item())
                    else:
                        fail_cnt+=1
            except good_epoch==False:
                epoch -=1


            md_logger.info("there are {} batches disgarded".format(fail_cnt))
            if good_epoch:
                if epoch > self.freezing_epochs:
                    md_logger.info(
                        f"Epoch {epoch}: teacher loss {t_loss.item()}, "
                        f"student loss {s_loss.item()}"
                    )
                else:
                    md_logger.info(
                        f"Epoch {epoch}: teacher is frozen (no loss computed), "
                        f"student loss {s_loss.item()}"
                    )
                wandb.log({'Teacher Train loss': np.mean(t_losses)}, step = epoch)
                wandb.log({'Student Train loss': np.mean(s_losses)}, step = epoch)
                # ----------------------------------
                #           evaluation
                # ----------------------------------
                trainmetrics = train_evaluator.eval(
                    self.student, train_pairs, train_labels, epoch
                )
                devmetrics = dev_evaluator.eval(self.student, dev_pairs, dev_labels, epoch)
                testmetrics = test_evaluator.eval(
                    self.student, test_pairs, test_labels, epoch
                )
                # ----------------------------------
                #           save records
                # ----------------------------------
                acc_train.append(trainmetrics[0])
                f1_train.append(trainmetrics[1])
                auc_train.append(trainmetrics[2])
                aupr_train.append(trainmetrics[3])

                acc_dev.append(devmetrics[0])
                f1_dev.append(devmetrics[1])
                auc_dev.append(devmetrics[2])
                aupr_dev.append(devmetrics[3])

                acc_test.append(testmetrics[0])
                f1_test.append(testmetrics[1])
                auc_test.append(testmetrics[2])
                aupr_test.append(testmetrics[3])

                # np.save(checkpoint_dir + "t_loss_train.npy", t_losses)
                # np.save(checkpoint_dir + "s_loss_train.npy", s_losses)
                # np.save(checkpoint_dir, "acc_train.npy", acc_train)
                # np.save(checkpoint_dir + "f1_train.npy", f1_train)
                # np.save(checkpoint_dir + "auc_train.npy", auc_train)
                # np.save(checkpoint_dir + "aupr_train.npy", aupr_train)

                # np.save(checkpoint_dir + "acc_dev.npy", acc_dev)
                # np.save(checkpoint_dir + "f1_dev.npy", f1_dev)
                # np.save(checkpoint_dir + "auc_dev.npy", auc_dev)
                # np.save(checkpoint_dir + "aupr_dev.npy", aupr_dev)

                # np.save(checkpoint_dir + "acc_test.npy", acc_test)
                # np.save(checkpoint_dir + "f1_test.npy", f1_test)
                # np.save(checkpoint_dir + "auc_test.npy", auc_test)
                # np.save(checkpoint_dir + "aupr_test.npy", aupr_test)

                record_dict["epoch"].append(epoch)

                record_dict["train_acc"].append(trainmetrics[0])
                record_dict["train_f1"].append(trainmetrics[1])
                record_dict["train_auc"].append(trainmetrics[2])
                record_dict["train_aupr"].append(trainmetrics[3])

                record_dict["dev_acc"].append(devmetrics[0])
                record_dict["dev_f1"].append(devmetrics[1])
                record_dict["dev_auc"].append(devmetrics[2])
                record_dict["dev_aupr"].append(devmetrics[3])

                record_dict["test_acc"].append(testmetrics[0])
                record_dict["test_f1"].append(testmetrics[1])
                record_dict["test_auc"].append(testmetrics[2])
                record_dict["test_aupr"].append(testmetrics[3])

                #target_metric = devmetrics[3]  # f1 for binary
                # save the best test for the convention
                target_metric = testmetrics[3]

                if target_metric > best_target_metric:
                    best_target_metric = target_metric  # new best metric
                # if save_model and testmetrics[2]>0.85 and testmetrics[3]>0.73:
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    torch.save(
                        self.student.state_dict(),
                        os.path.join(path, "student_state_dict.pth"),
                    )
                    md_logger.info(
                        "New best metric {:.6f} at epoch {}".format(
                            target_metric, best_epoch
                        )
                    )
            epoch+=1
        # save model weights at the end of training
        path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(
            self.student.state_dict(), os.path.join(path, "student_state_dict.pth"),
        )

        md_logger.info(
            "DevMetric {:.6f} at epoch {}. Current best DevMetric {:.6f} at "
            "epoch {}".format(target_metric, epoch, best_target_metric, best_epoch)
        )
        print(
            "Best DevMetric {:.6f} at epoch {}".format(best_target_metric, best_epoch)
        )
        return record_dict
