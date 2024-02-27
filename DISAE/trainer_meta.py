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
import pickle
from DISAE.utils import *



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

    def get_prot_repr_from_dataframe(self, df):
        prot_repr = list()
        is_str = isinstance(next(iter(self.uniprot2triplets.values())), str)

        for index, row in df.iterrows():
            protein_id = row['protein_id']
            if is_str:
                triplets = self.uniprot2triplets[protein_id].strip()
                n_triplets = len(triplets.split())
                if n_triplets < 254:
                    triplets += " [PAD]" * (254 - n_triplets)
                elif n_triplets > 254:
                    triplets = " ".join(triplets.strip().split()[:254])
                # md_logger.debug(f"triplets: {triplets}")
                # md_logger.debug(f"triplets length: {len(triplets.split())}")
                prot_repr.append(torch.tensor(self.berttokenizer.encode(triplets)))
            else:
                prot_repr.append(torch.tensor(self.uniprot2triplets[protein_id]))
        return torch.stack(prot_repr)


    def _prepare_chem_batch_for_nf(self, pairs):
        batch_chem_repr = self.get_chem_repr_from_pairs(pairs)
        return load_from_mol_tuple(batch_chem_repr)

    def _prepare_chem_batch_for_gin_from_dataframe(self, df):
        graphs = []
        #mol_is_graph = isinstance(next(iter(self.chemid2mol.values())), Data)
        mol_is_graph = True
        for index, row in df.iterrows():
            compound_id = row['compound_id']
            if mol_is_graph:
                graphs.append(self.chemid2mol[compound_id])
            else:
                graphs.append(mol_to_graph_data_obj_simple(self.chemid2mol[compound_id]))
        return Batch.from_data_list(graphs)



class Trainer_Meta(ModelRunner):
    def __init__(
        self,
        model=None,
        step=100,
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
        update_step=5,
        global_meta=30,
        global_eval_at=10,
        task_num=5
    ):
        self.batch_size = batch_size
        self.global_meta = global_meta
        self.update_step = update_step
        self.task_num = task_num
        self.global_eval_at = global_eval_at
        self.checkpoint_dir = ckpt_dir

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
    def meta_predict(self, batch_data_pertask, mode, detach=False):
        y_spt = torch.LongTensor(list(batch_data_pertask[mode]['activity'].values)).cuda()

        batch_prot_repr = self.get_prot_repr_from_dataframe(batch_data_pertask[mode])
        if (isinstance(batch_prot_repr, Variable) and torch.cuda.is_available()):
            batch_prot_repr = batch_prot_repr.cuda()

        batch_chem_embed = self._prepare_chem_batch_for_gin_from_dataframe(batch_data_pertask[mode])
        batch_chem_embed = batch_chem_embed.cuda()
        batch_input = {
            "protein": batch_prot_repr,
            "ligand": batch_chem_embed,
        }
        batch_logits = self.model(batch_input)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(batch_logits, y_spt)

        if detach:
            batch_logits = batch_logits.detach().cpu()
            y_spt = y_spt.detach().cpu()

        return batch_logits, y_spt, loss

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
        #model = self.model
        freeze_state = "freeze"
        parameters = list(self.model.parameters())
        optimizer = self._init_optimizer(parameters)
        scheduler = self._init_scheduler(optimizer)
        # ..............
        train_dict = load_pkl('/raid/home/yoyowu/DESSML/Data/Combined/activities/combined_all/02062024_combined_train_200_for_metaOOC.pk')
        
        trainscaf = list(train_dict.keys())

        #train_pairs, train_labels = load_data(edges, datatype="train")
        dev_pairs, dev_labels = load_data(dev_edges, datatype="dev")
        test_pairs, test_labels = load_data(test_edges, datatype="test")
        # ..............
        best_target_metric = -np.inf
        best_step = 0
        step = 0
        total_loss = 0
        batch_size = self.batch_size

        record_dict = defaultdict(list)


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
        for param in self.model.parameters():
            param.requires_grad = True


        # ----------------------------------
        #           training
        # ----------------------------------
        for step in range(1, self.global_meta + 1):
            # 5 cluster with 8 spt 4/4 pos/neg in one cluster
            batch_data = sample_minibatch_mixscaf(trainscaf, 5,1,4,1, train_dict)

            task_num = len(batch_data)
            losses_q_alltasks = 0

            task_metrics_all, class0_all_test, class1_all_test = [],[],[]
            # ---- mete update crosss tasks
            self.model.zero_grad()
            self.model.train()
            model_weights_meta_init = list(self.model.parameters())

            for t in range(task_num):

            # =========================core======================
                batch_data_pertask = batch_data[t]
                # --- -------------------support set updating--
                for s in range(self.update_step):

                    batch_logits, y_spt, loss = self.meta_predict(batch_data_pertask, 'spt')
                    grad = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
                    #grad = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], allow_unused=True)


                    fast_weights = []
                    for p in zip(grad, self.model.parameters()):
                        if type(p[0]) != type(None):
                            fast_weights.append(p[1] - 0.01 * p[0]) # meta update lr 0.01
                        else:
                            fast_weights.append(torch.zeros_like(p[1]))

                    for param_meta, param_update in zip(self.model.parameters(), fast_weights):
                        param_meta.data = param_update.data
             # --- -----------------query set get evaluated----
                batch_logits_q, y_qry, loss_q = self.meta_predict(batch_data_pertask, 'query', detach=True)
                losses_q_alltasks += loss_q
                task_metrics= meta_evaluate_binary_predictions(y_qry, batch_logits_q)
                task_metrics_all.append(task_metrics)

                # reset model param for next task
                for param_meta, param_init in zip(self.model.parameters(), model_weights_meta_init):
                    param_meta.data = param_init.data
            # =========================core======================
            # ---- mete update crosss tasks

            losses_q_metaupdate = losses_q_alltasks/self.task_num
            optimizer.zero_grad()
            losses_q_metaupdate.backward() #need to check what will be triggered by this
            optimizer.step()
            scheduler.step()
        # Calculate and log task average metrics
            avg_task_metrics = np.mean(task_metrics_all, axis=0)  # Assuming task_metrics_all is a list of lists
            #wandb.log({'Task Average ACC': avg_task_metrics[0]}, step=step)
            wandb.log({'Task Average F1': avg_task_metrics[0]}, step=step)
            wandb.log({'Task Average AUROC': avg_task_metrics[1]}, step=step)
            wandb.log({'Task Average AUPRC': avg_task_metrics[2]}, step=step)

            # ----------------------------------
            #           evaluation
            # ----------------------------------
            if step % self.global_eval_at ==0:
                print('------------------------training step: ', step)
                wandb.log({'Task Average F1': avg_task_metrics[0]}, step=step)
                wandb.log({'Task Average AUROC': avg_task_metrics[1]}, step=step)
                wandb.log({'Task Average AUPRC': avg_task_metrics[2]}, step=step)
                #trainmetrics = train_evaluator.eval(model, train_pairs, train_labels, step)
                devmetrics = dev_evaluator.eval(self.model, dev_pairs, dev_labels, step)
                testmetrics = test_evaluator.eval(self.model, test_pairs, test_labels, step)

                target_metric = testmetrics[3]  # use prauc

                if target_metric > best_target_metric:
                    best_target_metric = target_metric  # new best f1
                    best_step = step
                    path = os.path.join(self.checkpoint_dir, "best_auroc_dev")
                    #path = os.path.join(self.checkpoint_dir, "step_{0}".format(step))
                    if not os.path.exists(path):
                        os.mkdir(path)
                    torch.save(
                        self.model.state_dict(), os.path.join(path, "model_state_dict.pth")
                    )
                    md_logger.info(
                        "New best metric {:.6f} at step {}".format(
                            target_metric, best_step
                        )
                    )
        # save model weights at the end of training
        path = os.path.join(self.checkpoint_dir, "step_{0}".format(step))
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(
            self.model.state_dict(), os.path.join(path, "final_state_dict.pth"),
        )

        md_logger.info(
            "DevMetric {:.6f} at step {}. Current best DevMetric {:.6f} at "
            "step {}".format(target_metric, step, best_target_metric, best_step)
        )
        print(
            "Best DevMetric {:.6f} at step {}".format(best_target_metric, best_step)
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

