import os
import numpy as np

# ============================================================================
# IPM ZINC TRANSPLANTATION ENGINE (L01 HOLO-BUILDER)
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L01_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "l01_bft1_rank_001_v4.pdb")
WEM8_PDB = os.path.join(BASE_DIR, "benchmark_data", "category_1", "8WEM.pdb")
OUTPUT_PDB = os.path.join(BASE_DIR, "output_targets", "v4_multi_output", "L01_holo_receptor.pdb")

THREE_TO_ONE = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}

def parse_pdb(file_path, is_8wem=False):
    seq = ""
    coords = []
    zn_coords = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16] == " CA ":
                    res_name = line[17:20].strip()
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    seq += THREE_TO_ONE.get(res_name, 'X')
                    coords.append([x, y, z])
                elif is_8wem and line.startswith("HETATM") and " ZN " in line:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    zn_coords.append([x, y, z])
        return seq, np.array(coords), np.array(zn_coords)
    except Exception as e:
        print(f"[ERROR] Parsing {file_path}: {e}")
        return "", np.array([]), np.array([])

def align_and_transform(seq_l01, coords_l01, seq_8wem, coords_8wem, zn_8wem):
    n, m = len(seq_l01), len(seq_8wem)
    dp = np.zeros((n + 1, m + 1))
    for j in range(1, m + 1): dp[0, j] = -j * 2.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = 1.0 if seq_l01[i-1] == seq_8wem[j-1] and seq_l01[i-1] != 'X' else -2.0
            dp[i, j] = max(dp[i-1, j-1] + match, dp[i-1, j] - 1.0, dp[i, j-1] - 1.0)
            
    aligned_pairs = []
    i, j = np.unravel_index(np.argmax(dp), dp.shape)
    while i > 0 and j > 0:
        match = 1.0 if seq_l01[i-1] == seq_8wem[j-1] and seq_l01[i-1] != 'X' else -2.0
        if dp[i, j] == dp[i-1, j-1] + match:
            if match == 1.0: aligned_pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i, j] == dp[i-1, j] - 1.0: i -= 1
        else: j -= 1
        
    if len(aligned_pairs) < 10: return None
    
    P = np.array([coords_l01[i] for i, j in aligned_pairs[::-1]])
    Q = np.array([coords_8wem[j] for i, j in aligned_pairs[::-1]])
    
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    Pc = P - centroid_P
    Qc = Q - centroid_Q
    
    V, S, W = np.linalg.svd(np.dot(Qc.T, Pc))
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        S[-1] = -S[-1]; V[:, -1] = -V[:, -1]
        
    U = np.dot(V, W)
    
    zn_transformed = []
    for zn in zn_8wem:
        zn_centered = zn - centroid_Q
        zn_rot = np.dot(zn_centered, U)
        zn_final = zn_rot + centroid_P
        zn_transformed.append(zn_final)
        
    return np.array(zn_transformed)

def main():
    print("=======================================================")
    print("  IPM ZINC TRANSPLANTATION ENGINE (L01 HOLO-BUILDER)   ")
    print("=======================================================\n")
    
    if not os.path.exists(L01_PDB):
        print(f"[FATAL ERROR] Folded L01 PDB not found at {L01_PDB}.")
        print("Ensure ipm_hybrid_master_engine_cat_1.py has finished processing l01_bft1.fasta.")
        input("Press Enter to exit...")
        return
        
    print("[SYSTEM] Parsing Folded L01 (Apo-Receptor)...")
    seq_l01, coords_l01, _ = parse_pdb(L01_PDB)
    
    print("[SYSTEM] Parsing Crystal 8WEM (Calibration Anchor)...")
    if not os.path.exists(WEM8_PDB):
        print(f"[FATAL ERROR] Crystal structure 8WEM.pdb not found at {WEM8_PDB}.")
        input("Press Enter to exit...")
        return
        
    seq_8wem, coords_8wem, zn_8wem = parse_pdb(WEM8_PDB, is_8wem=True)
    
    if len(zn_8wem) == 0:
        print("[FATAL ERROR] No Zinc (ZN) atoms found in 8WEM.pdb.")
        input("Press Enter to exit...")
        return
        
    print(f"[SYSTEM] Identified {len(zn_8wem)} Zinc ion(s) in anchor. Engaging SVD spatial alignment...")
    
    zn_transformed = align_and_transform(seq_l01, coords_l01, seq_8wem, coords_8wem, zn_8wem)
    
    if zn_transformed is None:
        print("[FATAL ERROR] Spatial alignment failed. Insufficient sequence overlap.")
        input("Press Enter to exit...")
        return
        
    print("[SYSTEM] Spatial alignment complete. Transplanting Zinc coordinates into L01 model...")
    
    out_lines = []
    with open(L01_PDB, 'r') as f:
        for line in f:
            if line.startswith("TER") or line.startswith("END"): continue
            out_lines.append(line)
            
    out_lines.append("TER\n")
    for idx, zn_coord in enumerate(zn_transformed):
        x, y, z = zn_coord
        hetatm_line = f"HETATM{9000+idx:5d}  ZN   ZN A{900+idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          ZN\n"
        out_lines.append(hetatm_line)
        print(f"  -> Transplanted ZN {idx+1}: [{x:.3f}, {y:.3f}, {z:.3f}]")
        
    out_lines.append("END\n")
    
    with open(OUTPUT_PDB, 'w') as f:
        f.writelines(out_lines)
        
    print(f"\n[SUCCESS] Holo-Receptor generated: {OUTPUT_PDB}")
    print("[SYSTEM] Ready for Molecular Docking pipeline.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()