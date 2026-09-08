import os
import math
import numpy as np

# ============================================================================
# IPM L02 CHEMICAL SURGERY ENGINE (CYS253 -> CSO MODIFICATION)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "L02_rank_001_v4.pdb")
OUTPUT_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "L02_cso_receptor.pdb")

TARGET_RES_NUM = "253"

def main():
    print("=======================================================")
    print("  IPM L02 CHEMICAL SURGERY (CYS -> CSO OXIDATION)      ")
    print("=======================================================\n")

    if not os.path.exists(INPUT_PDB):
        print(f"[FATAL ERROR] Folded model not found at {INPUT_PDB}")
        return

    with open(INPUT_PDB, 'r') as f:
        lines = f.readlines()

    out_lines = []
    
    # לאיסוף הקואורדינטות לחישוב הוקטורי
    ca_coord = None
    cb_coord = None
    sg_coord = None
    sg_plddt = "96.59"
    sg_atom_serial = 0

    print(f"[SYSTEM] Scanning PDB for Cysteine at position {TARGET_RES_NUM}...")

    # מעבר ראשון: עדכון השם ל-CSO ושאיבת קואורדינטות בסיס
    for line in lines:
        if line.startswith("ATOM"):
            res_num = line[22:26].strip()
            
            if res_num == TARGET_RES_NUM:
                # שינוי שם השייר ל-CSO
                line = line[:17] + "CSO" + line[20:]
                
                atom_name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if atom_name == "CA":
                    ca_coord = np.array([x, y, z])
                elif atom_name == "CB":
                    cb_coord = np.array([x, y, z])
                elif atom_name == "SG":
                    sg_coord = np.array([x, y, z])
                    sg_plddt = line[60:66]
                    sg_atom_serial = int(line[6:11].strip())
                    
        out_lines.append(line)

    if sg_coord is None or cb_coord is None or ca_coord is None:
        print("[FATAL ERROR] Could not find CA, CB, or SG atoms for residue 253.")
        return

    print("[SYSTEM] CYS253 Located. Performing vector math for oxygen placement...")

    # חישוב וקטורי מרחבי - הוספת חמצן באוריינטציה ביו-פיזיקלית תקינה
    vec_cb_sg = sg_coord - cb_coord
    vec_cb_sg /= np.linalg.norm(vec_cb_sg)
    
    vec_ca_cb = cb_coord - ca_coord
    vec_ca_cb /= np.linalg.norm(vec_ca_cb)
    
    # מציאת וקטור ניצב למישור CA-CB-SG
    perp = np.cross(vec_ca_cb, vec_cb_sg)
    perp /= np.linalg.norm(perp)
    
    # מציאת וקטור בתוך המישור שניצב לקשר CB-SG
    in_plane = np.cross(perp, vec_cb_sg)
    
    # חישוב זווית (109.5 מעלות) ומרחק (1.6 אנגסטרם)
    angle = math.radians(109.5 - 180) # זווית יחסית להמשך הקשר
    distance = 1.60
    
    dir_od = vec_cb_sg * math.cos(angle) + in_plane * math.sin(angle)
    od_coord = sg_coord + dir_od * distance
    
    print(f"  -> Generated new OD atom coordinates: X:{od_coord[0]:.3f} Y:{od_coord[1]:.3f} Z:{od_coord[2]:.3f}")

    # יצירת השורה התקנית בפורמט PDB עבור אטום החמצן החדש
    new_serial = sg_atom_serial + 1
    od_line = (
        f"ATOM  {new_serial:5d}  OD  CSO A{TARGET_RES_NUM:>4s}    "
        f"{od_coord[0]:8.3f}{od_coord[1]:8.3f}{od_coord[2]:8.3f}"
        f"  1.00{sg_plddt}           O  \n"
    )

    # הזרקת השורה החדשה מיד אחרי אטום ה-SG
    final_pdb_lines = []
    for line in out_lines:
        final_pdb_lines.append(line)
        if line.startswith("ATOM") and line[22:26].strip() == TARGET_RES_NUM and line[12:16].strip() == "SG":
            final_pdb_lines.append(od_line)
            
    # תיקון מספור האטומים לאורך כל שאר הקובץ
    renumbered_lines = []
    current_serial = 1
    for line in final_pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
            prefix = line[:6]
            suffix = line[11:]
            renumbered_lines.append(f"{prefix}{current_serial:5d}{suffix}")
            current_serial += 1
        else:
            renumbered_lines.append(line)

    with open(OUTPUT_PDB, 'w') as f:
        f.writelines(renumbered_lines)

    print(f"\n[SUCCESS] CSO Modification complete! Modified receptor saved to:")
    print(f" -> {OUTPUT_PDB}")
    print("[SYSTEM] L02 Receptor is ready for Ligand Docking.")

if __name__ == "__main__":
    main()