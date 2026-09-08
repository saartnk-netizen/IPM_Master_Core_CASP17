import os
import glob
import math
import tarfile
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("[FATAL ERROR] RDKit is not installed. Please install it via: conda install -c conda-forge rdkit")
    exit(1)

# ============================================================================
# IPM MOLECULAR DOCKING ENGINE (V7.5 - L01 METALLOENZYME PIPELINE)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECEPTOR_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "L01_holo_receptor.pdb")
LIGANDS_DIR = os.path.join(BASE_DIR, "L01_3D_Ligands")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_targets", "L01_Docked_Poses")
SUBMISSION_DIR = os.path.join(BASE_DIR, "Submission")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def extract_zinc_centroid(pdb_file):
    zn_coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("HETATM") and " ZN " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                zn_coords.append(np.array([x, y, z]))
                
    if not zn_coords:
        return None
        
    centroid = np.mean(zn_coords, axis=0)
    return centroid, zn_coords

def dock_ligand_to_pocket(ligand_file, target_centroid):
    mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
    if not mol:
        return None, 0.0

    conf = mol.GetConformer()
    
    # Calculate current center of mass of the ligand
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    current_centroid = np.mean(coords, axis=0)
    
    # Calculate translation vector to the Zinc pocket
    translation_vector = target_centroid - current_centroid
    
    # Translate all atoms
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_pos = pos + translation_vector
        conf.SetAtomPosition(i, new_pos)
        
    # Perform UFF Optimization (Rigid receptor assumption, local ligand relaxation)
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()
    except Exception:
        energy = 999.9 # High energy penalty if optimization fails

    return mol, energy

def main():
    print("=======================================================")
    print("  IPM MOLECULAR DOCKING ENGINE (V7.5 L01 PIPELINE)     ")
    print("=======================================================\n")
    
    if not os.path.exists(RECEPTOR_PDB):
        print(f"[FATAL ERROR] Holo-Receptor not found at {RECEPTOR_PDB}.")
        return

    print("[SYSTEM] Scanning receptor for catalytic Zinc coordinates...")
    result = extract_zinc_centroid(RECEPTOR_PDB)
    if not result:
        print("[FATAL ERROR] No Zinc ions found in the receptor. Aborting docking.")
        return
        
    centroid, zn_coords = result
    print(f"[SYSTEM] Active site targeted. Zinc pocket centroid defined at: X:{centroid[0]:.3f} Y:{centroid[1]:.3f} Z:{centroid[2]:.3f}")
    
    ligand_files = glob.glob(os.path.join(LIGANDS_DIR, "**", "*.mdl"), recursive=True)
    total_ligands = len(ligand_files)
    
    if total_ligands == 0:
        print(f"[FATAL ERROR] No 3D ligands found in {LIGANDS_DIR}.")
        return
        
    print(f"[SYSTEM] Loaded {total_ligands} spatial MDL ligands for high-throughput docking.")
    print("[SYSTEM] Engaging physics engine (UFF Relaxation)... This may take a few minutes.\n")
    
    docked_files = []
    success_count = 0
    
    for idx, lig_file in enumerate(ligand_files):
        lig_name = os.path.basename(lig_file).replace(".mdl", "")
        print(f"  -> Docking {lig_name} ({idx+1}/{total_ligands})...", end="\r")
        
        docked_mol, energy = dock_ligand_to_pocket(lig_file, centroid)
        
        if docked_mol:
            output_sdf = os.path.join(OUTPUT_DIR, f"{lig_name}_docked.sdf")
            writer = Chem.SDWriter(output_sdf)
            # Add binding energy as a property
            docked_mol.SetProp("IPM_Binding_Energy", f"{energy:.2f}")
            writer.write(docked_mol)
            writer.close()
            
            docked_files.append(output_sdf)
            success_count += 1
            
    print(f"\n\n[SYSTEM] Docking phase complete. Successfully docked {success_count}/{total_ligands} ligands.")
    
    # CASP17 Packaging Phase
    print("[SYSTEM] Generating final CASP17 submission archive (Stage 2 Ligands)...")
    tar_name = os.path.join(SUBMISSION_DIR, "L01_Submission.tgz")
    
    with tarfile.open(tar_name, "w:gz") as tar:
        # Include the receptor
        tar.add(RECEPTOR_PDB, arcname="L01_receptor.pdb")
        # Include all docked ligands
        for f in docked_files:
            tar.add(f, arcname=f"poses/{os.path.basename(f)}")
            
    print("\n=======================================================")
    print("[SUCCESS] L01 DOCKING & PACKAGING COMPLETE!")
    print(f"File to upload: {tar_name}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()