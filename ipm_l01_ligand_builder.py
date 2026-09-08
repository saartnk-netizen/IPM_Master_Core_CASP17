import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem

# ============================================================================
# IPM LIGAND BUILDER (CASP17 LIGAND-SERIES PROTOCOL)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "L01_ligands") # חלץ את קובץ ה-tgz לכאן
OUTPUT_DIR = os.path.join(BASE_DIR, "L01_3D_Ligands")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def build_3d_ligand(smiles, output_path):
    # המרת SMILES לאובייקט מולקולה
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"    [!] Failed to parse SMILES: {smiles}")
        return False
        
    # הוספת אטומי מימן (קריטי לחישובי אנרגיה בעגינה מולקולרית)
    mol = Chem.AddHs(mol)
    
    # חישוב קואורדינטות תלת-ממדיות (Embedding)
    # מתכות ויחידים (כמו אבץ) לא צריכים שדה כוח קלאסי
    if mol.GetNumAtoms() > 1:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1:
            print(f"    [!] Failed to embed 3D coordinates for {smiles}")
            return False
            
        # אופטימיזציה גיאומטרית (מינימיזציה) לליגנד עם שדה כוח MMFF94
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"    [-] MMFF Optimization skipped: {e}")
    else:
        # טיפול באטום בודד (כמו Zn++)
        AllChem.Compute2DCoords(mol)
        
    # שמירה בפורמט התקני של התחרות (.mdl / .mol)
    Chem.MolToMolFile(mol, output_path)
    return True

def main():
    print("=======================================================")
    print("    IPM LIGAND EXTRACTOR & 3D BUILDER (RDKIT)          ")
    print("=======================================================\n")
    
    txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not txt_files:
        print(f"[FATAL] No .txt files found in {INPUT_DIR}. Please extract L01.smiles.tgz first.")
        return
        
    success_count = 0
    
    for txt_file in txt_files:
        target_id = os.path.basename(txt_file).replace('.smiles.txt', '').replace('.txt', '')
        print(f"-> Processing Target: {target_id}")
        
        target_out_dir = os.path.join(OUTPUT_DIR, target_id)
        os.makedirs(target_out_dir, exist_ok=True)
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        copy_counter = 1
        for line in lines:
            parts = line.strip().split()
            # דילוג על שורת הכותרת או שורות ריקות
            if not parts or parts[0] == "ID": 
                continue
                
            smiles = parts[2] if len(parts) >= 3 else None
            if not smiles:
                continue
                
            # תיוג ברור יותר עבור יון האבץ
            is_zinc = "Zn" in smiles
            suffix = "Zn" if is_zinc else f"copy_{copy_counter}"
            
            output_filename = os.path.join(target_out_dir, f"{target_id}_{suffix}.mdl")
            
            if build_3d_ligand(smiles, output_filename):
                print(f"    * Saved: {os.path.basename(output_filename)}")
                success_count += 1
                
            if not is_zinc:
                copy_counter += 1

    print("\n=======================================================")
    print(f"Extraction Complete. {success_count} 3D MDL files generated.")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    main()