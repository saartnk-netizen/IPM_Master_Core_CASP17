import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "benchmark_data", "L02.smiles") 
OUTPUT_DIR = os.path.join(BASE_DIR, "L02_3D_Ligands")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_3d_ligand(smiles, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1: return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    Chem.MolToMolFile(mol, output_path)
    return True

def main():
    print("=======================================================")
    print("    IPM L02 LIGAND BUILDER (V2 - MULTI-LIGAND FIX)     ")
    print("=======================================================\n")
    
    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.smiles.txt"))
    if not txt_files:
        print(f"[FATAL ERROR] No .smiles.txt files found in {INPUT_DIR}.")
        input("Press Enter to exit...")
        return
        
    success_count = 0
    for txt_file in txt_files:
        target_id = os.path.basename(txt_file).replace('.smiles.txt', '')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts or parts[0].upper() == "ID": 
                continue
            
            # Extract proper index to prevent overwrite
            lig_index = parts[0].replace("*", "").strip()
            if not lig_index: lig_index = "1"
            
            smiles = parts[2] if len(parts) >= 3 else None
            if not smiles: continue
                
            # ABSOLUTE FIX: Append lig_index to filename
            output_filename = os.path.join(OUTPUT_DIR, f"{target_id}_{lig_index}.mdl")
            
            if build_3d_ligand(smiles, output_filename):
                print(f"  -> Generated 3D coordinates for: {target_id} (Ligand {lig_index})")
                success_count += 1

    print(f"\n[SUCCESS] {success_count} 3D MDL ligands generated perfectly without overwriting.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()