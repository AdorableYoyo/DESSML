"""
Prediction with fine-tuned DISAE
"""

import os
import argparse
import logging
import json
import pickle as pk
from datetime import datetime

# rdkit need to be imported befor Torch, otherwise it sometimes causes ImportError
from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer
import sys
sys.path.append('/..')
# -------------
from microbiomemeta.predictor import Predictor
from microbiomemeta.utils import load_pairs_from_file
from DISAE.models import MolecularGraphCoupler
from DISAE.utils import (
    save_json,
    load_json,
    str2bool,
)


md_logger = logging.getLogger("prediction")


def set_hyperparameters():
    """ Setup hyperparameters

    Return:
        opt (Namespace): command line arguments.
    """
    parser = argparse.ArgumentParser("Predict with DISAE classifier")

    # args for ALBERT model
    parser.add_argument(
        "--protein_embedding_type",
        type=str,
        default="albert",
        help="albert, lstm are available options",
    )
    parser.add_argument(
        "--prot_feature_size",
        type=int,
        default=256,
        help="protein representation dimension",
    )
    parser.add_argument(
        "--prot_max_seq_len",
        type=int,
        default=256,
        help="maximum length of a protein sequence including special tokens",
    )
    # args for LSTM protein Embedding
    parser.add_argument(
        "--lstm_embedding_size",
        type=int,
        default=128,
        help="protein representation dimension for LSTM",
    )
    parser.add_argument(
        "--lstm_num_layers", type=int, default=3, help="num LSTM layers"
    )
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=64,
        help="protein representation dimension for LSTM",
    )
    parser.add_argument(
        "--lstm_out_size",
        type=int,
        default=128,
        help="protein representation dimension for LSTM",
    )
    # parameters for the chemical
    parser.add_argument(
        "--chem_conv_layer_sizes",
        type=lambda s: list(map(int, s.split(","))),
        default=[20, 20, 20, 20],
        help="Conv layers for chemicals",
    )
    parser.add_argument(
        "--chem_feature_size",
        type=int,
        default=128,
        help="chemical fingerprint dimension",
    )
    parser.add_argument(
        "--chem_degrees",
        type=lambda s: list(map(int, s.split(","))),
        default=[0, 1, 2, 3, 4, 5],
        help="Atomic connectivity degrees for chemical molecules",
    )
    parser.add_argument(
        "--state_dict_path",
        type=str,
        default="/raid/home/yoyowu/DESSML/trained_models/DTI/DTI_fold_1_DISAE_model_state_dict_0.pth",
        help="State dict path for trained DISAE model.",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="gin",
        help="Ligand embedding type. 'nf' (neuralfingerprint) or 'gin'.",
    )
    # args for Attentive Pooling
    parser.add_argument(
        "--ap_feature_size",
        type=int,
        default=64,
        help="attentive pooling feature dimension",
    )
    # args for model training and optimization
    parser.add_argument(
        "--datapath", default="/raid/home/yoyowu/DESSML/Data/DTI/DTI_test_x22102.tsv",
         help="Path to the data to be predicted."

    )
    parser.add_argument(
        "--prot2trp_path",
        type=str,
        default="/raid/home/yoyowu/DESSML/Data/Combined/proteins/triplets_in_my_data_set.pk",
        help="path of the protein id to triplets mapping json file.",
    )
    parser.add_argument(
        "--chm2smiles_path",
        type=str,
        default="/raid/home/yoyowu/DESSML/Data/Combined/chemicals/combined_compounds.pk",
        help="path of the chemical id to SMILES mapping tsv file.",
    )
    parser.add_argument(
        "--prediction_mode",
        default="binary",
        type=str,
        help="set to continuous and provide pretrained checkpoint",
    )
    parser.add_argument(
        "--batch", default=64, type=int, help="Batch size. (default 64)"
    )
    parser.add_argument(
        "--num_threads", default=8, type=int, help="Number of threads for torch"
    )
    parser.add_argument(
        "--log",
        default="INFO",
        type=str,
        help="Logging level. Set to DEBUG for more details.",
    )
    parser.add_argument(
        "--no_cuda",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Disables CUDA training.",
    )
    parser.add_argument(
        "--get_embeddings",
        dest='get_embeddings',
        action='store_true', help='whether to get embeddings from the model prediction')
    #parser.set_defaults(get_embeddings=True)
    opt = parser.parse_args()
    return opt


def set_folders(opt, save_folder="prediction_logs/"):
    """ Set up folders to save the prediction results.

    Args:
        save_folder (str): root path of the saving directory. A sub-directory will be
            created under the root directory to save the outputs. The name of the
            sub-directory is pred{current_time}.

    Return:
        logging_dir (str): path to the sub-directory.
    """
    # ---------- set folders ----------
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    md_logger.info(f"timestamp: {timestamp}")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    logging_dir = os.path.join(save_folder, f"pred{timestamp}")
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    return logging_dir


def set_logging(opt, format="%(asctime)-15s [%(levelname)s]: %(message)s"):
    """ Set up logging configurations.

    Args:
        opt (Namespace): arguments.
        format (str): logging format. Default: "%(asctime)-15s %(name)s [%(levelname)s]:
            %(message)s"
    """
    logging.basicConfig(format=format, level=getattr(logging, opt.log.strip().upper()))
    md_logger.info(opt)


def set_up_data(opt):
    """ Set up prediction data.

    Args:
        opt (NameSpace): auguments.

    Returns:
        pairs (dict): compound-protein pairs to be predicted.
        proteinid2triplets (dict): protein ID to triplets map.
        chemicalid2smiles (dict): compound ID to SMILES map.
        chemicalid2mol (dict): compound ID to molecular graphs map.
    """
    md_logger.info("Loading protein representations...")

    if opt.prot2trp_path.endswith(".json"):
        proteinid2triplets = load_json(opt.prot2trp_path)
    elif opt.prot2trp_path.endswith(".pk"):
        with open(opt.prot2trp_path, "rb") as f:
            proteinid2triplets = pk.load(f)

    md_logger.info(
        """
        Protein representations successfully loaded.
        Loading ligands.
        """
    )

    pairs = load_pairs_from_file(opt.datapath, sep="\t", header=False)
    md_logger.info("Protein-ligand pairs successfully loaded.")
    torch.set_num_threads(opt.num_threads)

    md_logger.info("Loading compounds mapping...")
    if opt.chm2smiles_path.endswith(".tsv"):
        chemicalid2smiles = {}
        with open(opt.chm2smiles_path, "r") as fin:
            for line in fin:
                line = line.strip().split("\t")
                chemid = line[1]
                smi = line[2]
                chemicalid2smiles[chemid] = smi
    elif opt.chm2smiles_path.endswith(".json"):
        with open(opt.chm2smiles_path) as f:
            chemicalid2smiles = json.load(f)
    elif opt.chm2smiles_path.endswith(".pk"):
        with open(opt.chm2smiles_path, "rb") as f:
            chemicalid2smiles = pk.load(f)

    if isinstance(next(iter(chemicalid2smiles.values())), Data):
        chemicalid2mol = chemicalid2smiles
    else:
        chemicalid2mol = {}
        for chemid, smiles in chemicalid2smiles.items():
            mol = MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(
                    f"Failed to convert {chemid}: {smiles} to RDKit Mol object."
                )
            chemicalid2mol[chemid] = mol
    md_logger.info("Chemical compounds mapping successfully loaded.")

    return (
        pairs,
        proteinid2triplets,
        chemicalid2smiles,
        chemicalid2mol,
    )


def set_up_models(opt):
    """ Set up the model.

    Args:
        opt (NameSpace): arguments.

    Returns:
        model: the MolecularGraphCoupler_MC model.
        berttokenizer: tokenizer for the BERT model.
    """
    berttokenizer = BertTokenizer.from_pretrained(
        "Data/DISAE_data/albertdata/vocab/pfam_vocab_triplets.txt"
    )

    model = MolecularGraphCoupler(
        protein_embedding_type=opt.protein_embedding_type,  # could be albert, LSTM,
        gnn_type=opt.gnn_type,  # could be "nf", "gin"
        prediction_mode=opt.prediction_mode,
        # protein features - albert
        albertconfig=None,
        tokenizer=berttokenizer,
        ckpt_path=None,
        frozen_list=[],
        # protein features - LSTM
        lstm_vocab_size=19688,
        lstm_embedding_size=opt.lstm_embedding_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_num_layers=opt.lstm_num_layers,
        lstm_out_size=opt.lstm_out_size,
        lstm_input_dropout_p=0,
        lstm_output_dropout_p=0,
        # chemical features
        conv_layer_sizes=opt.chem_conv_layer_sizes,
        output_size=opt.chem_feature_size,
        degrees=opt.chem_degrees,
        # attentive pooler features
        ap_hidden_size=opt.ap_feature_size,
        ap_dropout=0,
        gin_config=gin_config,
        get_embeddings = opt.get_embeddings
    )
    model.load_state_dict(torch.load(opt.state_dict_path))
    md_logger.info(f"Model weights loaded from {opt.state_dict_path}")
    if torch.cuda.is_available() and not opt.no_cuda:
        md_logger.info("Running on GPU ...")
        model = model.cuda()
    else:
        model = model.cpu()
        md_logger.info("Running on CPU...")
    return model, berttokenizer


if __name__ == "__main__":
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    opt = set_hyperparameters()
    set_logging(opt)
    logging_dir = set_folders(opt)
    config_path = os.path.join(logging_dir, "config.json")
    save_json(vars(opt), config_path)
    md_logger.info("model configurations saved to {}".format(config_path))
    if opt.gnn_type == "gin":
        gin_config = {
            "num_layer": 5,
            "emb_dim": 300,
            "JK": "last",
            "drop_ratio": 0,
            "checkpoint": None,
        }
    else:
        gin_config = None

    # ---------- set up trained models ----------
    model, berttokenizer = set_up_models(opt)

    # ---------- set up data ----------
    pairs, proteinid2triplets, chemicalid2smiles, chemicalid2mol = set_up_data(opt)

    # -------------------------------------------
    #      set up predictor
    # -------------------------------------------

    predictor = Predictor(
        model=model,
        berttokenizer=berttokenizer,
        batch_size=opt.batch,
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        protid2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
    )

    md_logger.info("Predictor initialized...")

    # -------------------------------------------
    #      predict
    # -------------------------------------------
    predictor.predict(pairs, logging_dir)
    md_logger.info("Prediction done.")
    md_logger.info(f"Log saved to {config_path}")
