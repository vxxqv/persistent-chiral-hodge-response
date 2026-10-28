"""Retrieve PubChem identities and generate a fixed, reproducible conformer panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import urlopen

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


# Fixed before obtaining coordinates or calculating any PCHR outcome.
PANEL = [
    ("L-alanine","amino acid"),("L-serine","amino acid"),
    ("L-valine","amino acid"),("L-leucine","amino acid"),
    ("L-isoleucine","amino acid"),("L-threonine","amino acid"),
    ("L-cysteine","amino acid"),("L-methionine","amino acid"),
    ("L-aspartic acid","amino acid"),("L-glutamic acid","amino acid"),
    ("L-asparagine","amino acid"),("L-glutamine","amino acid"),
    ("L-lysine","amino acid"),("L-arginine","amino acid"),
    ("L-histidine","amino acid"),("L-phenylalanine","amino acid"),
    ("L-tyrosine","amino acid"),("L-tryptophan","amino acid"),
    ("L-proline","amino acid"),
    ("(S)-lactic acid","hydroxy acid"),("(S)-mandelic acid","hydroxy acid"),
    ("(S)-ibuprofen","drug"),("(S)-naproxen","drug"),
    ("levodopa","catechol amino acid"),
]


def generate(out: Path, count: int = 8):
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for index,(name,family) in enumerate(PANEL):
        url=("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+quote(name,safe="")+
             "/property/IsomericSMILES,InChIKey/JSON")
        payload=None
        for attempt in range(3):
            try:
                with urlopen(url,timeout=45) as handle: payload=handle.read()
                break
            except Exception:
                if attempt==2: raise
                time.sleep(1+attempt)
        source=json.loads(payload)["PropertyTable"]["Properties"][0]
        smiles=source.get("SMILES",source.get("IsomericSMILES"))
        mol=Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol))!=1:
            raise ValueError(f"Invalid or disconnected identity: {name}")
        centers=Chem.FindMolChiralCenters(mol,includeUnassigned=True)
        if not centers or any(label=="?" for _,label in centers):
            raise ValueError(f"Ambiguous stereo identity: {name}")
        full=Chem.AddHs(mol)
        params=AllChem.ETKDGv3(); params.randomSeed=271828+index
        params.numThreads=1; params.pruneRmsThresh=-1
        ids=list(AllChem.EmbedMultipleConfs(full,numConfs=count,params=params))
        if len(ids)!=count: raise ValueError(f"Incomplete embedding: {name}")
        opt=AllChem.MMFFOptimizeMoleculeConfs(full,numThreads=1,maxIters=2000,mmffVariant="MMFF94s")
        if any(flag!=0 for flag,_ in opt):
            raise ValueError(f"Nonconverged MMFF geometry: {name}: {opt}")
        heavy=Chem.RemoveHs(full)
        features=[[a.GetAtomicNum(),a.GetDegree(),a.GetTotalNumHs()] for a in heavy.GetAtoms()]
        conformers=[]
        for conf_id,(flag,energy) in zip(ids,opt):
            conformers.append({"conformer_id":conf_id,"coordinates":heavy.GetConformer(conf_id).GetPositions().tolist(),
                               "mmff_status":flag,"mmff_energy_kcal_mol":energy})
        records.append({"name":name,"family":family,"pubchem_cid":source["CID"],
                        "inchikey":source["InChIKey"],"smiles":smiles,"source_url":url,
                        "source_sha256":hashlib.sha256(payload).hexdigest(),
                        "source_record":source,"atom_features":features,
                        "elements":[a.GetSymbol() for a in heavy.GetAtoms()],
                        "bonds":[[b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble()] for b in heavy.GetBonds()],
                        "stereocenters":[[int(i),label] for i,label in centers],
                        "seed":271828+index,"conformers":conformers})
        print(name,source["CID"],heavy.GetNumAtoms(),len(conformers),flush=True)
        time.sleep(.2)
    result={"generation":"RDKit ETKDGv3 followed by MMFF94s; explicit H during optimization; heavy atoms analysed",
            "rdkit_version":rdBase.rdkitVersion,"coordinate_unit":"angstrom",
            "requested_conformers":count,"source":"PubChem PUG REST compound identity, not PubChem3D coordinates",
            "retrieved_utc":datetime.now(timezone.utc).isoformat(),"records":records}
    (out/"conformers.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path("data"))
    parser.add_argument("--count",type=int,default=8)
    args=parser.parse_args(); generate(args.output,args.count)
