import logging
import time

import torch
from torch.autograd import Variable
from sklearn import metrics
import numpy as np
import wandb
from .trainer import ModelRunner


md_logger = logging.getLogger(__name__)


class Evaluator(ModelRunner):
    def __init__(
        self,
        chemid2smiles=None,
        chemid2mol=None,
        berttokenizer=None,
        uniprot2triplets=None,
        prediction_mode=None,
        protein_embedding_type=None,
        datatype="train",
        max_steps=1000,
        batch=64,
        shuffle=False,
    ):
        self.logger = logging.getLogger("DISAE.evaluator.Evaluator")
        self.chemid2smiles = chemid2smiles
        self.chemid2mol = chemid2mol
        self.berttokenizer = berttokenizer
        self.uniprot2triplets = uniprot2triplets
        self.prediction_mode = prediction_mode
        self.prottype = protein_embedding_type
        self.datatype = datatype
        self.max_steps = max_steps
        self.batch = batch
        self.shuffle = shuffle
        self.logger.info(
            "{} Evaluator for {} data initialized. Max {} steps for batch-size {}. \
        Shuffle {}".format(
                self.prediction_mode, datatype, max_steps, batch, shuffle
            )
        )

    def _get_chem_emb(self, model, pairs):
        if model.gnn_type == "nf":
            chem_emb = self._prepare_chem_batch_for_nf(pairs)
        elif model.gnn_type == "gin":
            chem_emb = self._prepare_chem_batch_for_gin(pairs)
        else:
            raise ValueError(
                f"The gnn_type of the model: {model.gnn_type} should be "
                f"'nf' or 'gin'."
            )
        return chem_emb

    def run_model(self, model, chem_emb, prot_repr, label, evaluate=True):
        if model.gnn_type == "gin":
            chem_emb = chem_emb.to("cuda:0")
        if isinstance(prot_repr, Variable) and torch.cuda.is_available():
            prot_repr = prot_repr.to("cuda:0")
        batch_input = {"protein": prot_repr, "ligand": chem_emb}
        with torch.no_grad():
            logits = model(batch_input)
        batch_labels = torch.tensor(label)
        batch_labels = batch_labels.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        self.logger.debug(
            "Evaluator-run_model: batch_labels {}, logits {}".format(
                batch_labels.shape, logits.shape
            )
        )
        if evaluate:
            acc, f1, auc, aupr = evaluate_binary_predictions(batch_labels, logits)
            return acc, f1, auc, aupr
        else:
            return batch_labels, logits

    def eval(self, model, pairs, labels, epoch):
        model.eval()

        datatype = self.datatype
        since = time.time()
        collected_logits = []
        collected_labels = []

        if len(pairs) <= self.batch:
            # sample size smaller than a batch
            prot_repr = self.get_prot_repr_from_pairs(pairs)
            chem_emb = self._get_chem_emb(model, pairs)
            metrics = self.run_model(model, chem_emb, prot_repr, labels, evaluate=True)
            eval_time = time.time() - since
            self.logger.info(
                "{:.2f} seconds for {} evaluation. Epoch {}".format(
                    eval_time, datatype, epoch
                )
            )
            print(
                "{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(
                    epoch, datatype, metrics[0], metrics[1], metrics[2], metrics[3]
                )
            )
            wandb.log({'{} ACC'.format(datatype): metrics[0]}, step=epoch)
            wandb.log({'{} F1'.format(datatype): metrics[1]}, step=epoch)
            wandb.log({'{} AUROC'.format(datatype): metrics[2]}, step=epoch)
            wandb.log({'{} AUPRC'.format(datatype): metrics[3]}, step=epoch)
            return metrics

        elif len(pairs) <= self.max_steps * self.batch:
            # evaluate all pairs if small enough
            maxsteps = int(np.ceil(float(len(pairs)) / float(self.batch)))
            for step in range(maxsteps):
                pairs_part = pairs[self.batch * step : self.batch * (step + 1)]
                if len(pairs_part) == self.batch:
                    labels_part = labels[self.batch * step : self.batch * (step + 1)]
                    prot_repr = self.get_prot_repr_from_pairs(pairs_part)
                    chem_emb = self._get_chem_emb(model, pairs_part)
                    # chem_repr,prot_repr = get_repr_from_pairs(pairs_part)
                    batch_labels, logits = self.run_model(
                        model, chem_emb, prot_repr, labels_part, evaluate=False
                    )
                    self.logger.debug(
                        "in-loop: batch_labels {}, logits {}".format(
                            batch_labels.shape, logits.shape
                        )
                    )
                    collected_logits.append(logits)
                    collected_labels.append(batch_labels)

        else:
            idxs = np.arange(len(pairs))
            np.random.shuffle(idxs)
            for step in range(self.max_steps):
                pairs_part = pairs[self.batch * step : self.batch * (step + 1)]
                labels_part = labels[self.batch * step : self.batch * (step + 1)]
                prot_repr = self.get_prot_repr_from_pairs(pairs_part)
                chem_emb = self._get_chem_emb(model, pairs_part)
                batch_labels, logits = self.run_model(
                    model, chem_emb, prot_repr, labels_part, evaluate=False
                )
                self.logger.debug(
                    "in-loop: batch_labels {}, logits {}".format(
                        batch_labels.shape, logits.shape
                    )
                )
                collected_logits.append(logits)
                collected_labels.append(batch_labels)

        collected_labels = np.concatenate(collected_labels, axis=0)
        collected_logits = np.concatenate(collected_logits, axis=0)
        if self.prediction_mode.lower() in ["binary"]:
            metrics = evaluate_binary_predictions(collected_labels, collected_logits)
        else:
            metrics = evaluate_continuous_predictions(
                collected_labels, collected_logits
            )

        eval_time = time.time() - since
        self.logger.info(
            "{:.2f} seconds for {} evaluation. Epoch {}".format(
                eval_time, datatype, epoch
            )
        )
        print(
            "{}\t{}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(
                epoch, datatype, metrics[0], metrics[1], metrics[2], metrics[3]
            )
        )
        wandb.log({'{} ACC'.format(datatype): metrics[0]}, step=epoch)
        wandb.log({'{} F1'.format(datatype): metrics[1]}, step=epoch)
        wandb.log({'{} AUROC'.format(datatype): metrics[2]}, step=epoch)
        wandb.log({'{} AUPRC'.format(datatype): metrics[3]}, step=epoch)
        return metrics


def evaluate_binary_predictions(label, predprobs):
    md_logger.debug("label {}, predprobs {}".format(label.shape, predprobs.shape))
    probs = np.array(predprobs)
    predclass = np.argmax(probs, axis=1)

    acc = metrics.accuracy_score(label, predclass)
    f1 = metrics.f1_score(label, predclass, average="weighted")
    fpr, tpr, thresholds = metrics.roc_curve(label, probs[:, 1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    prec, reca, thresholds = metrics.precision_recall_curve(
        label, probs[:, 1], pos_label=1
    )
    aupr = metrics.auc(reca, prec)
    return acc, f1, auc, aupr


def evaluate_continuous_predictions():
    pass
