import os
import argparse
import logging
import random
import json
import pickle as pk
from datetime import datetime
import numpy as np

# rdkit need to be imported befor Torch, otherwise it sometimes causes ImportError
from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.data import Data
from transformers import BertTokenizer
from transformers import AlbertConfig

from DISAE.models import MolecularGraphCoupler
from DISAE.utils import (
    load_edges_from_file,
    save_json,
    load_json,
)
from DISAE.evaluator import Evaluator
from microbiomemeta.utils import load_pairs_from_file
from microbiomemeta.trainers import MPLTrainer
import wandb

md_logger = logging.getLogger(__name__)


def set_hyperparameters():
    """ Set up hyperparameters.

    Return (NameSpace): arguments.
    """
    parser = argparse.ArgumentParser("Train DISAE based classifier")
    parser.add_argument(
        "--albert_checkpoint",
        type=str,
        default="experiment_logs/exp2023-02-14-22-54-57_chembl_njs16_47/epoch_21/model_state_dict.pth",
        help="Checkpoint path for pretrained albert.",
    )
    parser.add_argument(
        "--albert_pretrained_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for pretrained albert.",
    )
    # args for model training and optimization
    parser.add_argument("--train_datapath", default="Data/ChEMBL29/all_Chembl29.tsv")
    parser.add_argument("--dev_datapath", default="Data/NJS16/activities/Feb_2_23_dev_test/dev_47.tsv")
    parser.add_argument("--test_datapath", default="Data/NJS16/activities/Feb_2_23_dev_test/test_47.tsv")
    parser.add_argument("--ul_datapath", default="Data/new_unlabeled_data_2023/NJS16_unlabeled_0.001_seed27.tsv")
    parser.add_argument("--epoch", default=2, type=int)
    parser.add_argument(
        "--freezing_epochs",
        type=int,
        default=float("inf"),
        help="Number of epochs to freeze NLP model.",
    )
    parser.add_argument(
        "--teacher_lr", type=float, default=1e-5, help="Teacher learning rate"
    )
    parser.add_argument(
        "--student_lr", type=float, default=1e-5, help="Student learning rate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="label smoothing temperature"
    )
    parser.add_argument(
        "--student_lstm_dropout",
        type=float,
        default=0.3,
        help="LSTM layer output dropout ratio of the student model.",
    )
    parser.add_argument(
        "--student_ap_dropout",
        type=float,
        default=0.1,
        help="Attentive pooling layer dropout ratio of the student model.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        type=str,
        help="Logging level. Set to DEBUG for more details.",
    )
    parser.add_argument(
        "--config_file",
        default="micriobiomemeta/configs/mpl_finetune_config.json",
        help="General configs for Meta Pseudo Label fine-tuning.",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="save model."
    )
    parser.add_argument(
        "--blc_up", type=float, default=1.2,
        help="the upper bound of the balance ratio of the pseudo labels")
    parser.add_argument(
        "--blc_low", type=float, default=0.1,
        help="the lower bound of the balance ratio of the pseudo labels")
    parser.add_argument(
        "--exp_id", default=None
    )

    parser.add_argument("--test_run", action="store_true", help="Use testing setup.")
    opt = parser.parse_args()
    # load default configs
    with open("microbiomemeta/configs/mpl_finetune_config.json", "r") as f:
        config_map = json.load(f)
    for k, v in config_map.items():
        setattr(opt, k, v)

    return opt


def set_folders(opt, save_folder="experiment_logs/"):
    """ Set up folders to save the fine-tuning results.

    Args:
        opt (Namespace): arguments.
        save_folder (str): root path of the save directory. A sub-directory will be
            created under the root directory to save the outputs. The name of the
            sub-directory is exp{current_time}.

    Return:
        checkpoint_dir (str): path to the sub-directory.
    """
    # ---------- set folders ----------
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    md_logger.info(f"timestamp: {timestamp}")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    checkpoint_dir = "{}/exp{}/".format(save_folder, timestamp)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    seed = opt.random_seed
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    return checkpoint_dir


def set_logging(opt, format="%(asctime)-15s %(name)s [%(levelname)s]: %(message)s"):
    """ Set up logging configurations.

    Args:
        opt (Namespace): arguments.
        format (str): logging format. Default: "%(asctime)-15s %(name)s [%(levelname)s]:
            %(message)s"
    """
    logging.basicConfig(
        format=format,
        level=getattr(logging, opt.log.upper()),
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    md_logger.info(opt)


def set_up_pretrained_albert(opt, data_path, vocab, config, checkpoint, vocab_size):
    """ Set up the pretrained ALBERT model. Add ALBERT's vocabulary path, checkpoint
    path, and vocabulary size to the argument Namespace.

    Args:
        opt (Namespace): arguments.
        data_path (str): path to the albert data root directory.
        vocab (str): vocabulary filename.
        config (str): config filename.
        checkpoint (str): checkpoint path for pretrained albert.
        vocab_size (int): size of the vocabulary.

    Return (AlbertConfig): the configuration for the ALBERT model based on the config
    file.
    """
    opt.albertdatapath = data_path
    opt.albertvocab = os.path.join(opt.albertdatapath, vocab)
    opt.albertconfig = os.path.join(opt.albertdatapath, config)
    opt.albert_pretrained_checkpoint = checkpoint
    opt.lstm_vocab_size = vocab_size
    return AlbertConfig.from_pretrained(opt.albertconfig)


class DataSetter:
    """ Class for setting up data for Meta Pseudo Label training.

    Attributes:
        proteinid2triplets (dict): map from protein IDs to their triplet
            representations.
        chemicalid2smiles (dict): map from chemical compound IDs to SMILES.
        chemicalid2mol (dict): map from chemical compound IDs to molecular graphs.
    """

    def _load_triplets(self, opt):
        """ Helper method for the __init__. Load protein triplets from json or pickle
        file.
        """
        if opt.prot2trp_path.endswith(".json"):
            triplets = load_json(opt.prot2trp_path)
        elif opt.prot2trp_path.endswith(".pk"):
            with open(opt.prot2trp_path, "rb") as f:
                triplets = pk.load(f)
        return triplets

    def _load_chem_mapping(self):
        """ Helper method for the __init__. Load chemical compounds to SMILES map from
        file. The file can in tsv, json, or pickle format. If chemical compound to graph
        file exists, load it, otherwise, generat graphs from SMILES.

        Returns:
            chemicalid2smiles (dict): chemical compounds to SMILES map.
            chemicalid2mol (dict): chemical compounds to molecular graph map.
        """
        md_logger.info("Loading compounds mapping...")
        if self.opt.test_run:
            self.opt.chm2smiles_path = "/raid/home/yangliu/MicrobiomeMeta/for_test/Combined_activity/n_sampled_chem_map.pk"
        if self.opt.chm2smiles_path.endswith(".tsv"):
            chemicalid2smiles = {}
            with open(self.opt.chm2smiles_path, "r") as fin:
                for line in fin:
                    line = line.strip().split("\t")
                    chemid = line[1]
                    smi = line[2]
                    chemicalid2smiles[chemid] = smi
        elif self.opt.chm2smiles_path.endswith(".json"):
            with open(self.opt.chm2smiles_path) as f:
                chemicalid2smiles = json.load(f)
        elif self.opt.chm2smiles_path.endswith(".pk"):
            with open(self.opt.chm2smiles_path, "rb") as f:
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
        return chemicalid2smiles, chemicalid2mol

    def __init__(self, opt):
        """
        Args:
            opt (NameSpace): arguments.
        """
        self.opt = opt
        md_logger.info("Loading protein representations...")
        self.proteinid2triplets = self._load_triplets(opt)
        self.chemicalid2smiles, self.chemicalid2mol = self._load_chem_mapping()
        md_logger.info("Protein representations loaded.")

    def load_data(self):
        """ Load labeled data.

        Returns:
            train_edges (dict): keys are compound id + protein id, values are labels for
            keys.
            train_chem_ids (list): id list of chemical compounds.
            train_prot_ids (list): id list of proteins.
            dev_edges (dict): the same as train_edges.
            dev_chem_ids (list): the same as train_chem_ids.
            dev_prot_ids (list): the same as train_prot_ids.
            test_edges (dict): the same as train_edges.
            test_chem_ids (list): the same as train_chem_ids.
            test_prot_ids (list): the same as train_prot_ids.
        """

        md_logger.info(
            """
            Protein representations successfully loaded.
            Loading protein-ligand interactions.
            """
        )
        if self.opt.test_run:
            train_edges, train_chem_ids, train_prot_ids = load_edges_from_file(
                "for_test/Combined_activity/train.tsv", sep="\t", header=False
            )
            dev_edges, dev_chem_ids, dev_prot_ids = load_edges_from_file(
                "for_test/Combined_activity/dev.tsv", sep="\t", header=False
            )
            test_edges, test_chem_ids, test_prot_ids = load_edges_from_file(
                "for_test/Combined_activity/test.tsv", sep="\t", header=False
            )
            md_logger.info(
                "Using testing run setup. Testing data loaded. "
                "Remove --test_run tag to disabel test run."
            )
        else:
            train_edges, train_chem_ids, train_prot_ids = load_edges_from_file(
                self.opt.train_datapath, sep="\t", header=False
            )
            dev_edges, dev_chem_ids, dev_prot_ids = load_edges_from_file(
                self.opt.dev_datapath, sep="\t", header=False
            )
            test_edges, test_chem_ids, test_prot_ids = load_edges_from_file(
                self.opt.test_datapath, sep="\t", header=False
            )
            md_logger.info("Labeled protein-ligand interactions successfully loaded.")

        return (
            train_edges,
            train_chem_ids,
            train_prot_ids,
            dev_edges,
            dev_chem_ids,
            dev_prot_ids,
            test_edges,
            test_chem_ids,
            test_prot_ids,
        )

    def load_ul_data(self):
        """ Load unlabeled data.

        Returns:
            edges (dict): map contains chemical compound and protein pairs.
        """
        if self.opt.test_run:
            edges = load_pairs_from_file(
                "for_test/Combined_activity/test_ul_pairs.tsv", sep="\t", header=False
            )
        else:
            edges = load_pairs_from_file(self.opt.ul_datapath, sep="\t", header=False)
            md_logger.info("Unlabeled protein-ligand pairs successfully loaded.")
        return edges


def set_up_finetuning_models(
    opt,
    albertconfig,
    gin_config,
    tokenizer,
    ckpt_path=None,
    lstm_dropout=0,
    ap_dropout=0,
):
    """ Set up the model.

    Args:
        opt (NameSpace): arguments.
        albertconfig (AlbertConfig): the configuration object for the transformer model.
        gin_config (dict): the configuration for the gin model.
        tokenizer (Tokenizer): the Tokenizer object.
        ckpt_path (str): path to the directory where pre-trained weights are saved.
        lstm_dropout (float): the dropout rate for the LSTM model. Default is 0.
        ap_dropout (float): the dropout rate for the attentive pooling layer. Default is
            0.
    Returns:
        model: the MolecularGraphCoupler model.
    """
    model = MolecularGraphCoupler(
        protein_embedding_type=opt.protein_embedding_type,  # could be albert, LSTM,
        gnn_type=opt.gnn_type,  # could be "nf", "gin"
        prediction_mode=opt.prediction_mode,
        # protein features - albert
        albertconfig=albertconfig,
        tokenizer=tokenizer,
        ckpt_path=opt.albert_pretrained_checkpoint,
        frozen_list=[],
        # protein features - LSTM
        lstm_vocab_size=19688,
        lstm_embedding_size=opt.lstm_embedding_size,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_num_layers=opt.lstm_num_layers,
        lstm_out_size=opt.lstm_out_size,
        lstm_input_dropout_p=0,
        lstm_output_dropout_p=lstm_dropout,
        # chemical features
        conv_layer_sizes=opt.chem_conv_layer_sizes,
        output_size=opt.chem_feature_size,
        degrees=opt.chem_degrees,
        # attentive pooler features
        ap_hidden_size=opt.ap_feature_size,
        ap_dropout=ap_dropout,
        gin_config=gin_config,
    )
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    md_logger.info(f"Model weights loaded from {ckpt_path}")
    if torch.cuda.is_available():
        md_logger.info("Moving model to GPU ...")
        model = model.cuda()
    else:
        model = model.cpu()
        md_logger.info("Running on CPU...")
    return model


if __name__ == "__main__":
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    opt = set_hyperparameters()
    set_logging(opt)
    checkpoint_dir = set_folders(opt)
    torch.set_num_threads(opt.num_threads)

    # ---------- set up fine-tuning models ----------
    if opt.gnn_type == "gin":
        gin_config = {
            "num_layer": 5,
            "emb_dim": 300,
            "JK": "last",
            "drop_ratio": 0.2,
            "checkpoint": opt.gin_checkpoint,
        }
    else:
        gin_config = None
    md_logger.info("Setting up models...")
    berttokenizer = BertTokenizer.from_pretrained(
        "Data/DISAE_data/albertdata/vocab/pfam_vocab_triplets.txt"
    )
    teacher_config = set_up_pretrained_albert(
        opt,
        data_path="Data/DISAE_data/albertdata/",
        vocab="vocab/pfam_vocab_triplets.txt",
        config="albertconfig/albert_config_tiny_google.json",
        checkpoint=opt.albert_pretrained_checkpoint,
        vocab_size=19688,
    )
    student_config = set_up_pretrained_albert(
        opt,
        data_path="Data/DISAE_data/albertdata/",
        vocab="vocab/pfam_vocab_triplets.txt",
        config="albertconfig/albert_config_tiny_google.json",
        checkpoint=None,
        vocab_size=19688,
    )
    t_model = set_up_finetuning_models(
        opt, teacher_config, gin_config, berttokenizer, opt.albert_checkpoint
    )
    s_model = set_up_finetuning_models(
        opt,
        student_config,
        gin_config,
        berttokenizer,
        lstm_dropout=opt.student_lstm_dropout,
        ap_dropout=opt.student_ap_dropout,
    )
    md_logger.info("Models successfully set up.")

    config_path = os.path.join(checkpoint_dir, "config.json")
    save_json(vars(opt), config_path)
    md_logger.info("model configurations saved to {}".format(config_path))

    wandb.init(project="microbio_meta",config=opt)
    wandb.watch(s_model, log="all")
    wandb.watch(t_model, log="all")
    # ---------- set up data ----------
    data_setter = DataSetter(opt)
    proteinid2triplets = data_setter.proteinid2triplets
    chemicalid2smiles = data_setter.chemicalid2smiles
    chemicalid2mol = data_setter.chemicalid2mol
    edges, _, _, dev_edges, _, _, test_edges, _, _ = data_setter.load_data()
    ul_edges = data_setter.load_ul_data()

    # ------- set up trainer and evaluator --------
    mpl_trainer = MPLTrainer(
        teacher=t_model,
        student=s_model,
        tokenizer=berttokenizer,
        epoch=opt.epoch,
        batch_size=opt.batch,
        ckpt_dir=checkpoint_dir,
        optimizer=opt.optimizer,
        l2=opt.l2,
        teacher_lr=opt.teacher_lr,
        student_lr=opt.student_lr,
        scheduler=opt.scheduler,
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        prot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        freezing_epochs=opt.freezing_epochs,
        temperature=opt.temperature
    )

    train_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="train",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=True,
    )

    dev_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="dev",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )

    test_evaluator = Evaluator(
        chemid2smiles=chemicalid2smiles,
        chemid2mol=chemicalid2mol,
        berttokenizer=berttokenizer,
        uniprot2triplets=proteinid2triplets,
        prediction_mode=opt.prediction_mode,
        protein_embedding_type=opt.protein_embedding_type,
        datatype="test",
        max_steps=opt.max_eval_steps,
        batch=opt.batch,
        shuffle=False,
    )
    md_logger.info("Train and Dev evaluators initialized.\nStart training...")

    # ------- training and evaluating -------
    record_dict = mpl_trainer.train(
        edges=edges,
        unlabeled_edges=ul_edges,
        train_evaluator=train_evaluator,
        dev_edges=dev_edges,
        dev_evaluator=dev_evaluator,
        test_edges=test_edges,
        test_evaluator=test_evaluator,
        checkpoint_dir=checkpoint_dir,
        blc_up=opt.blc_up,
        blc_low=opt.blc_low,
        save_model=opt.save_model
    )
    record_path = os.path.join(checkpoint_dir, "training_record.json")
    save_json(record_dict, record_path)
    # md_logger.info("Training record saved to {}".format(record_path))
    # print("Training record saved to {}".format(record_path))
