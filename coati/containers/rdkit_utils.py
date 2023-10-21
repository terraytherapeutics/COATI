#
# Any useful rdkit functions are sequestered to this file.
# to keep minimal dependence on it. Instead it imports stuff from here
#
import functools
import random
from operator import itemgetter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    PandasTools,
    rdMolDescriptors,
)
from rdkit.Chem.AllChem import (
    EmbedMolecule,
    EmbedMultipleConfs,
    GetMorganFingerprintAsBitVect,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from rdkit.Chem.SaltRemover import SaltRemover


def works_on_smiles(raise_on_failure: bool):
    """
    converts any function mapping a mol to another mol (or any output)
    to a function that also works with smiles.

    I had too much fun with this.
    """

    def decorator(mol_func):
        @functools.wraps(mol_func)
        def wrapped_func(*args, **kwargs):
            if isinstance(args[0], str):
                if mol := Chem.MolFromSmiles(args[0]):
                    new_args = list(args)
                    new_args[0] = mol
                    try:
                        results = mol_func(*new_args, **kwargs)
                    except Exception as Ex:
                        if raise_on_failure:
                            raise Ex
                        else:
                            print(f"Exception: {Ex} for smiles: {args[0]}")
                            return None
                    # try to convert back to smiles...
                    if isinstance(results, Chem.Mol):
                        return Chem.MolToSmiles(results)
                    elif isinstance(results, tuple):
                        return tuple(
                            Chem.MolToSmiles(res) if isinstance(res, Chem.Mol) else res
                            for res in results
                        )
                    else:
                        return results
                else:
                    if raise_on_failure:
                        raise ValueError(f"{args[0]} could not be converted to mol.")
                    else:
                        return None
            else:
                return mol_func(*args, **kwargs)

        return wrapped_func

    return decorator


def rdkit_version():
    return rdkit.__version__


def canon_smiles(s):
    try:
        m = Chem.MolFromSmiles(s)
        if not m is None:
            Chem.Kekulize(m)
            return Chem.MolToSmiles(m)
        else:
            return "BAD_SMILES"
    except:
        return "BAD_SMILES"


@works_on_smiles(raise_on_failure=True)
def sim_mol(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Simple wrapper over ECFP4 tanimoto similarity. Returns a float between 0 and 1.
    """
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def identical_canonsmi(smi1: str, smi2: str, use_chiral: int = 1) -> bool:
    return Chem.CanonSmiles(smi1, useChiral=use_chiral) == Chem.CanonSmiles(
        smi2, useChiral=use_chiral
    )


@works_on_smiles(raise_on_failure=True)
def draw_mol(mol: Chem.Mol, size=(300, 300)):
    return Draw.MolToImage(mol, size=size)


def permute_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    ans = list(range(mol.GetNumAtoms()))
    random.shuffle(ans)
    nm = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(nm, canonical=False)


def draw_smi_grid(
    smis: List[str], mols_per_row=5, sub_img_size=(300, 300), legends=None
):
    return Draw.MolsToGridImage(
        [Chem.MolFromSmiles(smi) for smi in smis],
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
    )


def disable_logger():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


@works_on_smiles(raise_on_failure=True)
def mol_to_morgan(
    mol: Chem.Mol,
    radius: int = 3,
    n_bits: int = 2048,
    chiral: bool = False,
    features: bool = False,
) -> np.ndarray:
    return np.frombuffer(
        GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=chiral,
            useFeatures=features,
        )
        .ToBitString()
        .encode(),
        "u1",
    ) - ord("0")


@works_on_smiles(raise_on_failure=False)
def mol_to_atoms_coords(
    m: Chem.Mol,
    hydrogenate: bool = True,
    adj_matrix: bool = False,
    do_morgan: bool = False,
    optimize: bool = False,
    numConfs: int = 1,
    numThreads: int = 1,
):
    """
    Simple wrapper around RDKit ETKDG embedding + MMFF opt.
    """

    m3 = Chem.AddHs(m) if hydrogenate else m
    if optimize and hydrogenate:
        try:
            EmbedMultipleConfs(
                m3,
                randomSeed=0xF00D,
                numConfs=numConfs,
                pruneRmsThresh=0.125,
                ETversion=1,
                numThreads=numThreads,
            )
            opt = MMFFOptimizeMoleculeConfs(
                m3, mmffVariant="MMFF94s", numThreads=numThreads, maxIters=10000
            )
            opt = np.array(opt)
            converged = opt[:, 0] == 0
            lowest_eng_conformer = np.argmin(opt[converged][:, 1])
            lowest_energy = opt[converged][lowest_eng_conformer, 1]
            best_conf = np.arange(opt.shape[0])[converged][lowest_eng_conformer]
            c0 = m3.GetConformer(id=int(best_conf))
        except Exception as Ex:
            EmbedMolecule(m3, randomSeed=0xF00D)
            c0 = m3.GetConformers()[-1]
            lowest_energy = None
    else:
        EmbedMolecule(m3, randomSeed=0xF00D)
        c0 = m3.GetConformers()[-1]
    coords = c0.GetPositions()
    atoms = np.array([X.GetAtomicNum() for X in m3.GetAtoms()], dtype=np.uint8)

    to_return = [atoms, coords]

    if adj_matrix:
        to_return.append(Chem.GetAdjacencyMatrix(m3))

    # NOT using the relaxed/confgen'd molecule with HS - rdkit
    # is surprisingly sensitive to this.
    if do_morgan:
        to_return.append(mol_to_morgan(m, radius=3, n_bits=2048, chiral=False))

    if optimize:
        to_return.append(lowest_energy)

    return tuple(to_return)


def read_sdf(sdf: Any) -> pd.DataFrame:
    return PandasTools.LoadSDF(sdf, smilesName="SMILES")


@works_on_smiles(raise_on_failure=False)
def mol_standardize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """removes salts, takes largest mol fragment, neutralizes."""
    # salt removal
    salt_remover = SaltRemover()
    res_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    # # uncharged mol

    if res_mol.GetNumAtoms():
        # largest component
        frag_list = list(Chem.GetMolFrags(res_mol, asMols=True))
        frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_mw_list.sort(key=itemgetter(0), reverse=True)
        # neutralize fragment
        if len(frag_mw_list) > 0:
            return Uncharger().uncharge(frag_mw_list[0][1])
        return None
    else:
        print(f'Failed salt removal: "{Chem.MolToSmiles(mol)}"')
        return None


@works_on_smiles(raise_on_failure=False)
def mol_properties(mol: Chem.Mol) -> Dict[str, Any]:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "FractionCSP3": Lipinski.FractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
        "RingCount": Lipinski.RingCount(mol),
        "MolLogP": Crippen.MolLogP(mol),
    }
