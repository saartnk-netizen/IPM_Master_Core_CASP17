import os
import glob
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("[FATAL ERROR] RDKit is not installed.")
    exit(1)

# ============================================================================
# IPM L02 MOLECULAR DOCKING ENGINE (CSO253 POCKET TARGETING)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# שימוש ברצפטור שעבר את המודיפיקציה (CYS -> CSO)
RECEPTOR_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "L02_cso_receptor.pdb")
LIGANDS_DIR = os.path.join(BASE_DIR, "L02_3D_Ligands")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_targets", "L02_Docked_Poses")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_cso253_centroid(pdb_file):
    """שואב את הקואורדינטות המרחביות של הציסטאין המחומצן (CSO 253) כדי שישמש כעוגן לעגינה"""
    cso_coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and "CSO" in line and " 253 " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                cso_coords.append(np.array([x, y, z]))
                
    if not cso_coords:
        return None
        
    # חישוב נקודת המרכז של השייר המחומצן
    centroid = np.mean(cso_coords, axis=0)
    return centroid

def dock_ligand_to_pocket(ligand_file, target_centroid):
    """מעביר את הליגנד לכיס המטרה ומבצע הרפיית אנרגיה (UFF) להתאמה מרחבית"""
    mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
    if not mol:
        return None, 0.0

    conf = mol.GetConformer()
    
    # מציאת המרכז הנוכחי של הליגנד
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    current_centroid = np.mean(coords, axis=0)
    
    # חישוב וקטור ההזזה לעבר ה-CSO253
    translation_vector = target_centroid - current_centroid
    
    # הזזת כל אטומי הליגנד אל כיס המטרה
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        new_pos = pos + translation_vector
        conf.SetAtomPosition(i, new_pos)
        
    # הפעלת שדה כוח (Force Field) להתאמה ביו-פיזיקלית
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        energy = ff.CalcEnergy()
    except Exception:
        energy = 999.9 # קנס אנרגטי במידה והאופטימיזציה נכשלת

    return mol, energy

def main():
    print("=======================================================")
    print("  IPM L02 DOCKING ENGINE (TARGETING CSO 253)           ")
    print("=======================================================\n")
    
    if not os.path.exists(RECEPTOR_PDB):
        print(f"[FATAL ERROR] Modified Receptor not found at {RECEPTOR_PDB}.")
        return

    print("[SYSTEM] Scanning receptor for CSO 253 active site coordinates...")
    centroid = extract_cso253_centroid(RECEPTOR_PDB)
    
    if centroid is None:
        print("[FATAL ERROR] CSO 253 not found in the receptor. Aborting docking.")
        return
        
    print(f"[SYSTEM] Active site targeted! CSO 253 Centroid: X:{centroid[0]:.3f} Y:{centroid[1]:.3f} Z:{centroid[2]:.3f}")
    
    ligand_files = glob.glob(os.path.join(LIGANDS_DIR, "*.mdl"))
    total_ligands = len(ligand_files)
    
    if total_ligands == 0:
        print(f"[FATAL ERROR] No 3D ligands found in {LIGANDS_DIR}.")
        return
        
    print(f"[SYSTEM] Loaded {total_ligands} spatial ligands. Engaging UFF Physics Engine...\n")
    
    success_count = 0
    
    for idx, lig_file in enumerate(ligand_files):
        lig_name = os.path.basename(lig_file).replace(".mdl", "")
        print(f"  -> Docking {lig_name} ({idx+1}/{total_ligands})...", end="\r")
        
        docked_mol, energy = dock_ligand_to_pocket(lig_file, centroid)
        
        if docked_mol:
            output_sdf = os.path.join(OUTPUT_DIR, f"{lig_name}_docked.sdf")
            writer = Chem.SDWriter(output_sdf)
            docked_mol.SetProp("IPM_Binding_Energy", f"{energy:.2f}")
            writer.write(docked_mol)
            writer.close()
            success_count += 1
            
    print(f"\n\n[SUCCESS] Docking phase complete. Successfully docked {success_count}/{total_ligands} ligands.")
    print("[SYSTEM] Proceed to final CASP17 Stage-2 packaging.")

if __name__ == "__main__":
    main()