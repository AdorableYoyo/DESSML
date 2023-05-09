import logging
import os
import time
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch_geometric.data import Data, Batch
import numpy as np
from wandb import wandb
from .fingerprint.graph import load_from_mol_tuple
from microbiomemeta.data.utils.converters import mol_to_graph_data_obj_simple


md_logger = logging.getLogger(__name__)


def load_data(edges_dict, datatype="train"):
    # load training pairs with labels
    # edges: list of tuples (inchikey,uniprotID)
    # labels: list of float activity values for each edge
    count = 0
    count_skipped = 0
    labels = []
    edges = []
    chems = []
    prots = []
    for cp in edges_dict.keys():
        chem, prot = cp.strip().split("\t")
        chems.append(chem)
        prots.append(prot)
        count += 1
        labels.append(edges_dict[cp])
        edges.append((chem, prot))
    chems = list(set(chems))
    prots = list(set(prots))
    md_logger.info(
        "Total {} chemicals, {} proteins, {} activities loaded for {} data. "
        "{} chemicals skipped for non-Mol conversion".format(
            len(chems), len(prots), len(labels), datatype, count_skipped
        )
    )
    return edges, labels


class ModelRunner:
    def __init__(self):
        """ The base model for Trainer and Evaluator
        """

    def get_chem_repr_from_pairs(self, pairs):
        """ Functions to process protein sequence.
        for protein sequence in triplets
        """
        chem_repr = [
            (self.chemid2smiles[pair[0]], self.chemid2mol[pair[0]]) for pair in pairs
        ]
        return chem_repr

    def get_prot_repr_from_pairs(self, pairs):
        prot_repr = list()
        is_str = isinstance(next(iter(self.uniprot2triplets.values())), str)

        for pair in pairs:
            if is_str:
                triplets = self.uniprot2triplets[pair[1]].strip()
                n_triplets = len(triplets.split())
                if n_triplets < 254:
                    triplets += " [PAD]" * (254 - n_triplets)
                elif n_triplets > 254:
                    triplets = " ".join(triplets.strip().split()[:254])
                else:
                    pass
                # md_logger.debug(f"triplets: {triplets}")
                # md_logger.debug(f"triplets length: {len(triplets.split())}")
                prot_repr.append(torch.tensor(self.berttokenizer.encode(triplets)))
            else:
                prot_repr.append(torch.tensor(self.uniprot2triplets[pair[1]]))
        return torch.stack(prot_repr)

    def _prepare_chem_batch_for_nf(self, pairs):
        batch_chem_repr = self.get_chem_repr_from_pairs(pairs)
        return load_from_mol_tuple(batch_chem_repr)

    def _prepare_chem_batch_for_gin(self, pairs):
        graphs = []
        mol_is_graph = isinstance(next(iter(self.chemid2mol.values())), Data)
        for pair in pairs:
            if mol_is_graph:
                graphs.append(self.chemid2mol[pair[0]])
            else:
                graphs.append(mol_to_graph_data_obj_simple(self.chemid2mol[pair[0]]))
        return Batch.from_data_list(graphs)


class Trainer(ModelRunner):
    def __init__(
        self,
        model=None,
        epoch=100,
        batch_size=32,
        ckpt_dir="./temp/",
        optimizer="adam",
        l2=1e-3,
        lr=1e-5,
        scheduler="cosineannealing",
        prediction_mode=None,
        chemid2smiles=None,
        protein_embedding_type=None,
        uniprot2triplets=None,
        chemid2mol=None,
        berttokenizer=None,
        freezing_epochs=0,
    ):
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.train_epoch = epoch
        self.optimizer = optimizer
        self.l2 = l2
        self.lr = lr
        self.freezing_epochs = freezing_epochs
        self.scheduler = scheduler
        if self.scheduler.lower() == "cyclic":
            self.optimizer = "sgd"
            md_logger.info(
                "CyclicLR scheduler is used. Optimizer is set to {}".format(
                    self.optimizer.upper()
                )
            )
        self.model = model
        self.prediction_mode = prediction_mode
        if self.prediction_mode is None:
            raise AttributeError(
                "Prediction mode must be specified (binary or continuous)"
            )
        self.prottype = protein_embedding_type
        if self.prottype is None:
            raise AttributeError(
                "Protein embedding type must be specified ( LSTM, or ALBERT)"
            )
        self.uniprot2triplets = uniprot2triplets
        self.chemid2smiles = chemid2smiles
        self.chemid2mol = chemid2mol
        self.berttokenizer = berttokenizer
        if self.model is None:
            raise AttributeError("model not provided")
        if self.uniprot2triplets is None:
            raise AttributeError("dict uniprot2triplets not provided")
        if self.chemid2mol is None:
            raise AttributeError("dict ikey2mol not provided")
        if self.berttokenizer is None:
            raise AttributeError("Bert tokenizer not provided")

    def _init_optimizer(self, parameters):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.l2)
            md_logger.info(
                "Optimizer {}, LR {}, Weight Decay {}".format(
                    self.optimizer, self.lr, self.l2
                )
            )
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.l2)
            md_logger.info(
                "Optimizer {}, LR {}, Weight Decay {}".format(
                    self.optimizer, self.lr, self.l2
                )
            )
        return optimizer

    def _init_scheduler(self, optimizer):
        if self.scheduler == "cosineannealing":
            tmax = 10
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tmax
            )
            md_logger.info("Scheduler {}, T_max {}".format(self.scheduler, tmax))
        elif self.scheduler == "cyclic":
            max_lr = self.lr
            base_lr = self.lr * 0.01
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=max_lr
            )
            md_logger.info(
                "Scheduler {}, base_lr {:.8f}, max_lr {:.8f} ".format(
                    self.scheduler, base_lr, max_lr
                )
            )
        return scheduler

    def _show_gradients(self, model, freeze_state):
        md_logger.debug(
            f"Frozen list: {model.proteinEmbedding.frozen_list}, "
            f"freeze state: {freeze_state}"
        )
        if (
            model.proteinEmbedding.frozen_list is not None
            and len(model.proteinEmbedding.frozen_list) > 0
        ):
            layer = model.proteinEmbedding.frozen_list[0]
            module = list(model.proteinEmbedding.albert.modules())[layer - 1]
            for param in module.parameters():
                md_logger.debug(f"Gradient of layer {layer}'s parameters: {param.grad}")
                break

    def train(
        self,
        edges,
        train_evaluator,
        dev_edges,
        dev_evaluator,
        test_edges,
        test_evaluator,
        checkpoint_dir,
    ):
        # ----------------------------------
        #    set up data/parameters/models
        # ----------------------------------
        model = self.model
        freeze_state = "freeze"
        parameters = list(self.model.parameters())
        optimizer = self._init_optimizer(parameters)
        scheduler = self._init_scheduler(optimizer)
        # ..............
        train_pairs, train_labels = load_data(edges, datatype="train")
        dev_pairs, dev_labels = load_data(dev_edges, datatype="dev")
        test_pairs, test_labels = load_data(test_edges, datatype="test")
        # ..............
        best_target_metric = -np.inf
        best_epoch = 0
        step = 0
        total_loss = 0
        batch_size = self.batch_size
        batch_per_epoch = int(np.ceil(len(train_labels) / batch_size))
        record_dict = defaultdict(list)
        print("Epoch\tData\tAcc\tF1\tAUC\tAUPR")

        loss_train = []
        acc_train = []
        f1_train = []
        auc_train = []
        aupr_train = []

        acc_dev = []
        f1_dev = []
        auc_dev = []
        aupr_dev = []

        acc_test = []
        f1_test = []
        auc_test = []
        aupr_test = []

        # ----------------------------------
        #           training
        # ----------------------------------
        for epoch in range(1, self.train_epoch + 1):
            model.train()
            if epoch > self.freezing_epochs:
                freeze_state = "free"
                model.proteinEmbedding._unfreeze_albert()
            train_data_idxs = list(range(len(train_labels)))
            np.random.shuffle(train_data_idxs)
            epoch_loss_total = 0
            epoch_loss = []
            batch_prep_time = 0
            batch_train_time = 0
            batch_optim_time = 0
            md_logger.info("Epoch {0} started".format(epoch))
            # ----- one epoch
            for batch_ in range(batch_per_epoch):
                stime = time.time()
                choices = train_data_idxs[
                    batch_ * batch_size : (batch_ + 1) * batch_size
                ]
                if len(choices) == batch_size:
                    batch_labels = torch.tensor(
                        [train_labels[idx] for idx in choices]
                    ).cuda()
                    # ----------------------------------
                    #           process input
                    # ----------------------------------
                    batch_train_pairs = [train_pairs[idx] for idx in choices]
                    batch_prot_repr = self.get_prot_repr_from_pairs(batch_train_pairs)
                    if model.gnn_type == "nf":
                        batch_chem_embed = self._prepare_chem_batch_for_nf(
                            batch_train_pairs
                        )
                    elif model.gnn_type == "gin":
                        batch_chem_embed = self._prepare_chem_batch_for_gin(
                            batch_train_pairs
                        )
                        batch_chem_embed = batch_chem_embed.to("cuda:0")
                    else:
                        raise ValueError(
                            f"The gnn_type of the model: {model.gnn_type} should be "
                            f"'nf' or 'gin'."
                        )
                    # ----- move variables to GPU
                    if (
                        isinstance(batch_prot_repr, Variable)
                        and torch.cuda.is_available()
                    ):
                        batch_prot_repr = batch_prot_repr.cuda()

                    batch_input = {
                        "protein": batch_prot_repr,
                        "ligand": batch_chem_embed,
                    }
                    batch_prep_t = time.time() - stime
                    batch_prep_time += batch_prep_t
                    stime = time.time()
                    # ----------------------------------
                    #       get prediction score
                    # ----------------------------------
                    batch_logits = model(batch_input)
                    batch_train_t = time.time() - stime
                    batch_train_time += batch_train_t
                    stime = time.time()
                    # ----------------------------------
                    #            loss
                    # ----------------------------------

                    loss_fn = torch.nn.CrossEntropyLoss()
                    batch_labels = batch_labels.long()
                    loss = loss_fn(batch_logits, batch_labels)
                    epoch_loss.append(loss.detach().cpu().numpy())
                    optimizer.zero_grad()
                    loss.backward()
                    self._show_gradients(model, freeze_state)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    batch_optim_t = time.time() - stime
                    batch_optim_time += batch_optim_t
                    total_loss += loss.item()
                    epoch_loss_total += loss.item()

                    md_logger.debug(
                        "Epoch {}: BatchPrepTime {:.1f}, BatchTrainTime {:.1f}, "
                        "BatchOptimTime {:.1f}".format(
                            epoch, batch_prep_time, batch_train_time, batch_optim_time
                        )
                    )


            md_logger.info("Epoch {}: Loss {}".format(epoch, loss.item()))
            wandb.log({"Loss": loss.item()}, step=epoch)
            # ----------------------------------
            #           evaluation
            # ----------------------------------
            trainmetrics = train_evaluator.eval(model, train_pairs, train_labels, epoch)
            devmetrics = dev_evaluator.eval(model, dev_pairs, dev_labels, epoch)
            testmetrics = test_evaluator.eval(model, test_pairs, test_labels, epoch)
            # ----------------------------------
            #           save records
            # ----------------------------------
            loss_train.append(epoch_loss)
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

            # np.save(checkpoint_dir + "loss_train.npy", loss_train)
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
            record_dict["total_loss"].append(loss.item())

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

            target_metric = devmetrics[3]  # use prauc

            if target_metric > best_target_metric:
                best_target_metric = target_metric  # new best f1
                best_epoch = epoch
                path = os.path.join(self.checkpoint_dir, "epoch_{0}".format(epoch))
                if not os.path.exists(path):
                    os.mkdir(path)
                torch.save(
                    model.state_dict(), os.path.join(path, "model_state_dict.pth")
                )
                md_logger.info(
                    "New best metric {:.6f} at epoch {}".format(
                        target_metric, best_epoch
                    )
                )
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
        return (
            record_dict,
            loss_train,
            f1_train,
            auc_train,
            aupr_train,
            f1_dev,
            auc_dev,
            aupr_dev,
        )
