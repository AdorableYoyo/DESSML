import os
import logging
import time

import torch
from torch.autograd import Variable
import numpy as np
from DISAE.trainer import ModelRunner


md_logger = logging.getLogger(__name__)


class Predictor(ModelRunner):
    def __init__(
        self,
        model=None,
        berttokenizer=None,
        batch_size=32,
        chemid2smiles=None,
        chemid2mol=None,
        protid2triplets=None,
        prediction_mode=None,
        protein_embedding_type=None,
    ):
        """ The Class for applying predictions.

        Args:
            model: the model for running the prediction.
            berttokenizer (Tokenizer): the tokenizer instance for tokenizing the input
                protein sequence.
            batch_size (int): batch size. Default is 32. This number will NOT affect
                the final prediction results, use a number your VRAM can handle.
            chemid2smiles (dict): map from chemical compound IDs to SMILES.
            chemid2mol (dict): map from chemical compound IDs to molecular graphs.
            protid2triplets (dict): map from protein IDs to triplets.
            prediction_mode (str): "binary" or "continuous".
            protein_embedding_type (str): "LSTM" or "ALBERT".
        """
        self.batch_size = batch_size
        self.model = model
        self.prediction_mode = prediction_mode
        if self.prediction_mode is None:
            raise AttributeError(
                "Prediction mode must be specified (binary or continuous)"
            )
        self.prottype = protein_embedding_type
        if self.prottype is None:
            raise AttributeError(
                "Protein embedding type must be specified (LSTM, or ALBERT)"
            )
        self.uniprot2triplets = protid2triplets
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

    def predict(self, pairs, save_dir, log_frequency=10):
        """ Method to run the prediction.

        Args:
            pairs (dict): dictionary contains compound-protein pairs for prediction.
            save_dir (str): path to save the prediction results.
            log_frequency (int): frequency (# of batches) to print the prediction status
                on console. Default is 10.
        """
        # ----------------------------------
        #    set up data/parameters/models
        # ----------------------------------
        model = self.model
        # ..............
        batch_size = self.batch_size
        n_steps = int(np.ceil(len(pairs) / batch_size))
        # ----------------------------------
        #           prediction
        # ----------------------------------
        model.eval()
        keys = list(pairs.keys())
        logging.info("Prediction started")
        batch_pred_time = 0
        predictions = None
        for batch_ in range(n_steps):
            stime = time.time()
            choices = keys[batch_ * batch_size : (batch_ + 1) * batch_size]
            batch_pairs = [pairs[k] for k in choices]
            # ----------------------------------
            #           process input
            # ----------------------------------
            batch_prot_repr = self.get_prot_repr_from_pairs(batch_pairs)
            if model.gnn_type == "nf":
                batch_chem_embed = self._prepare_chem_batch_for_nf(batch_pairs)
            elif model.gnn_type == "gin":
                batch_chem_embed = self._prepare_chem_batch_for_gin(batch_pairs)
                batch_chem_embed = batch_chem_embed.to("cuda:0")
            else:
                raise ValueError(
                    f"The gnn_type of the model: {model.gnn_type} should be "
                    f"'nf' or 'gin'."
                )
            # ----- move variables to GPU
            if isinstance(batch_prot_repr, Variable) and torch.cuda.is_available():
                batch_prot_repr = batch_prot_repr.cuda()

            batch_input = {
                "protein": batch_prot_repr,
                "ligand": batch_chem_embed,
            }
            # ----------------------------------
            #       get prediction score
            # ----------------------------------
            batch_logits = model(batch_input)
            if predictions is None:
                predictions = batch_logits.detach().cpu().numpy()
            else:
                try:
                    logits = batch_logits.detach().cpu().numpy()
                    predictions = np.concatenate([predictions, logits], axis=0)
                except Exception:
                    print("error here")
                    return predictions
            batch_pred_t = time.time() - stime
            batch_pred_time += batch_pred_t
            stime = time.time()
            if (batch_ + 1) % log_frequency == 0 or (batch_ + 1) == n_steps:
                md_logger.info(
                    f"batch: {batch_ + 1}/{n_steps}, time: {batch_pred_t:.2f}, "
                    f"total time: {batch_pred_time:.2f}"
                )
        try:
            np.save(os.path.join(save_dir, "predict_logits.npy"), predictions)
        except Exception as e:
            print("error in saving predictions with numpy")
            # return predictions
            md_logger.error(e)

        return predictions
