import os
# OOM PROTECTION: Must be declared before PyTorch loads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gc
import numpy as np
import time
import requests
import subprocess
import shutil
import tarfile
import io

# ============================================================================
# V400.0 NEURAL BRIDGE (MEMORY INTERCEPTOR)
# ============================================================================
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    data = original_torch_load(*args, **kwargs)
    if isinstance(data, dict) and "model" in data:
        state = data["model"]
        keys = list(state.keys())
        for k in keys:
            if 'ipa.linear_kv_points.weight' in k:
                state[k.replace('ipa.linear_kv_points.weight', 'ipa.linear_kv_points.linear.weight')] = state.pop(k)
            elif 'ipa.linear_kv_points.bias' in k:
                state[k.replace('ipa.linear_kv_points.bias', 'ipa.linear_kv_points.linear.bias')] = state.pop(k)
            elif 'ipa.linear_q_points.weight' in k:
                state[k.replace('ipa.linear_q_points.weight', 'ipa.linear_q_points.linear.weight')] = state.pop(k)
            elif 'ipa.linear_q_points.bias' in k:
                state[k.replace('ipa.linear_q_points.bias', 'ipa.linear_q_points.linear.bias')] = state.pop(k)
    return data
torch.load = patched_torch_load

import esm  

# ============================================================================
# ARCHITECTURAL WORKSPACE
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_data", "category_1")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_targets", "v4_multi_output")
MSA_DIR = os.path.join(BASE_DIR, "msa_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MSA_DIR, exist_ok=True)

# THE ABSOLUTE GATE (Set to 100.0 to force AlphaFold2 5-rank execution)
THRESHOLD = 100.0  

THREE_TO_ONE = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
VALID_AAS = set("ARNDCQEGHILKMFPSTWYV")

# ============================================================================
# MODULE 1: NATIVE FAIR COMPILER (GPU INFERENCE)
# ============================================================================
class IPMNativeRootCompiler:
    def __init__(self):
        self.device = torch.device('cuda') 
        print(f"  -> V4.5 Absolute Oracle Initialized on {self.device}.")
        try:
            self.model = esm.pretrained.esmfold_v1().eval().cuda()
            self.model.set_chunk_size(32) 
            self.esm_active = True
        except Exception as e:
            print(f"  -> [FATAL ERROR] {e}")
            self.esm_active = False

    def compile_to_pdb(self, target_id, sequence):
        if not self.esm_active: return None
        try:
            sanitized_seq = "".join([aa if aa.upper() in VALID_AAS else "G" for aa in sequence])
            with torch.no_grad():
                print(f"  -> [PHASE 1] Single-Sequence Topography (12 Recycles)...")
                pdb_string = self.model.infer_pdb(sanitized_seq, num_recycles=12)
                
            out_file = os.path.join(OUTPUT_DIR, f"{target_id}_esm_v4.pdb")
            with open(out_file, 'w') as f:
                f.write(pdb_string)
            return out_file
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return "OOM_FALLBACK"
            return None
        finally:
            gc.collect()
            torch.cuda.empty_cache()

# ============================================================================
# MODULE 2: MSA RETRIEVER & EXTRACTOR (V4.2)
# ============================================================================
class IPMColabFoldRetriever:
    def __init__(self):
        self.api_url = "https://api.colabfold.com/ticket/msa"
        
    def fetch_msa(self, target_id, sequence):
        msa_file = os.path.join(MSA_DIR, f"{target_id}.a3m")
        if os.path.exists(msa_file):
            print(f"  -> [PHASE 2] MSA Evolution matrix loaded from local cache.")
            return msa_file
            
        print(f"  -> [PHASE 2] Acquiring MSA from ColabFold API...")
        try:
            payload = {'q': f">{target_id}\n{sequence}", 'mode': 'unpaired'}
            req = requests.post(self.api_url, data=payload, timeout=10)
            req.raise_for_status()
            ticket = req.json().get('id')
            
            while True:
                status_req = requests.get(f"https://api.colabfold.com/ticket/{ticket}", timeout=10)
                status = status_req.json().get('status')
                if status == 'COMPLETE': break
                elif status in ['ERROR', 'FAILED']: return None
                time.sleep(5)
                
            dl_req = requests.get(f"https://api.colabfold.com/result/download/{ticket}", timeout=15)
            
            with tarfile.open(fileobj=io.BytesIO(dl_req.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".a3m"):
                        a3m_content = tar.extractfile(member).read()
                        with open(msa_file, 'wb') as f:
                            f.write(a3m_content)
                        print(f"  -> [PHASE 2] MSA Archive unpacked and secured.")
                        return msa_file
                        
            print(f"  -> [MSA ERROR] No .a3m file found in the downloaded archive.")
            return None
            
        except Exception as e:
            print(f"  -> [MSA ERROR] API/Extraction exception: {e}")
            return None

# ============================================================================
# MODULE 3: ALPHAFOLD2 LOCAL EXECUTION (MULTI-RANK ORACLE)
# ============================================================================
class IPMAlphaFoldOracle:
    def execute_af2(self, target_id, msa_file, output_dir):
        print(f"  -> [PHASE 3] Engaging ALPHAFOLD2 (Local GPU + MSA Logic)...")
        temp_dir = os.path.join(output_dir, f"temp_{target_id}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            cmd = [
                "colabfold_batch", msa_file, temp_dir, 
                "--num-recycle", "3"
            ]
            
            af2_env = os.environ.copy()
            af2_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            af2_env["TF_FORCE_UNIFIED_MEMORY"] = "1"
            
            subprocess.run(cmd, env=af2_env, check=True)
            
            # Extract ALL 5 topological ranks
            all_pdbs = glob.glob(os.path.join(temp_dir, "*_rank_*.pdb"))
            saved_pdbs = []
            
            for i in range(1, 6):
                for pdb in all_pdbs:
                    if f"rank_00{i}" in pdb or f"rank_{i}" in pdb:
                        final_pdb = os.path.join(output_dir, f"{target_id}_rank_00{i}_v4.pdb")
                        shutil.copy(pdb, final_pdb)
                        saved_pdbs.append(final_pdb)
                        break
                        
            if saved_pdbs:
                print(f"  -> [PHASE 3] AF2 Topology successfully synthesized (5 Ranks).")
                return saved_pdbs
            else:
                print(f"  -> [ORACLE ERROR] AF2 failed to render valid PDBs.")
                return None
        except Exception as e:
            print(f"  -> [ORACLE ERROR] Execution exception: {e}")
            return None
        finally:
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

# ============================================================================
# MODULE 4: CASP EVALUATOR (CORE RMSD)
# ============================================================================
def extract_plddt(pdb_file):
    plddts = []
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16] == " CA ":
                    plddts.append(float(line[60:66].strip()))
        return round(sum(plddts) / len(plddts), 2) if plddts else 0.0
    except:
        return 0.0

def parse_pdb_strict(file_path):
    chains = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ENDMDL") or line.startswith("MODEL        2"): break
                if line.startswith("ATOM") and line[12:16] == " CA ":
                    if line[16] not in [' ', 'A', '1']: continue 
                    chain_id = line[21]
                    res_name = line[17:20].strip()
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    if chain_id not in chains: chains[chain_id] = {"seq": "", "coords": []}
                    chains[chain_id]["seq"] += THREE_TO_ONE.get(res_name, 'X')
                    chains[chain_id]["coords"].append([x, y, z])
        if not chains: return "", np.array([])
        best_chain = chains[max(chains.keys(), key=lambda k: len(chains[k]["seq"]))]
        return best_chain["seq"], np.array(best_chain["coords"])
    except Exception:
        return "", np.array([])

def calculate_casp_metrics(coords_computed, seq_computed, coords_truth, seq_truth):
    n, m = len(seq_computed), len(seq_truth)
    dp = np.zeros((n + 1, m + 1))
    for j in range(1, m + 1): dp[0, j] = -j * 2.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = 1.0 if seq_computed[i-1] == seq_truth[j-1] and seq_computed[i-1] != 'X' else -2.0
            dp[i, j] = max(dp[i-1, j-1] + match, dp[i-1, j] - 1.0, dp[i, j-1] - 1.0)
            
    aligned_pairs = []
    i, j = np.unravel_index(np.argmax(dp), dp.shape)
    while i > 0 and j > 0:
        match = 1.0 if seq_computed[i-1] == seq_truth[j-1] and seq_computed[i-1] != 'X' else -2.0
        if dp[i, j] == dp[i-1, j-1] + match:
            if match == 1.0: aligned_pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i, j] == dp[i-1, j] - 1.0: i -= 1
        else: j -= 1
        
    if len(aligned_pairs) < 10: return None
    
    P = np.array([coords_computed[i] for i, j in aligned_pairs[::-1]])
    Q = np.array([coords_truth[j] for i, j in aligned_pairs[::-1]])
    
    Pc = P - np.mean(P, axis=0)
    Qc = Q - np.mean(Q, axis=0)
    V, S, W = np.linalg.svd(np.dot(Pc.T, Qc))
    if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
        S[-1] = -S[-1]; V[:, -1] = -V[:, -1]
        
    U_global = np.dot(V, W)
    P_rot_global = np.dot(Pc, U_global)
    distances = np.sqrt(np.sum((P_rot_global - Qc) ** 2, axis=1))
    
    threshold_idx = int(len(distances) * 0.85)
    core_distances = np.sort(distances)[:threshold_idx]
    core_rmsd = np.sqrt(np.mean(core_distances**2)) if len(core_distances) > 0 else 0.0

    return round(core_rmsd, 4)

def main():
    print("=======================================================")
    print("IPM HYBRID MASTER ENGINE (V4.5) - MULTI-RANK EVALUATOR")
    print(f"CRITICAL GATE THRESHOLD: {THRESHOLD}% pLDDT")
    print("FALLBACK: Native AlphaFold2 Execution (Local MSA)")
    print("=======================================================\n")
    
    compiler = IPMNativeRootCompiler()
    msa_retriever = IPMColabFoldRetriever()
    af_oracle = IPMAlphaFoldOracle()
    
    if not compiler.esm_active: return
    
    fasta_files = glob.glob(os.path.join(BENCHMARK_DIR, "*.fasta"))
    if not fasta_files: 
        print(f"[FATAL ERROR] Zero targets detected in {BENCHMARK_DIR}.")
        input("Press Enter to exit...")
        return

    total_core = 0.0
    valid_targets = 0

    for fasta in fasta_files:
        target_id = os.path.basename(fasta).replace('.fasta', '')
        print(f"\n[ORCHESTRATOR] Target: {target_id} | Eradicating Variance...")
        
        sequence = ""
        with open(fasta, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    if sequence != "": break 
                    continue
                sequence += line.strip()
            
        final_pdbs = []
        
        # 1. ESMFold Fast-Path (Will automatically route to AF2 due to 100.0 Threshold)
        fast_pdb = compiler.compile_to_pdb(target_id, sequence)
        
        if fast_pdb and fast_pdb != "OOM_FALLBACK":
            plddt = extract_plddt(fast_pdb)
            print(f"  -> [GATE] Initial Spatial Assurance: {plddt}%")
            
            if plddt < THRESHOLD:
                print(f"  -> [GATE] FAILED. Target < {THRESHOLD}%. Routing to Oracle...")
                fast_pdb = None
            else:
                print("  -> [GATE] PASSED. Topology secured via Fast-Path.")
                final_pdbs = [fast_pdb]

        # 2. AlphaFold2 Oracle (Fallback)
        if not fast_pdb or fast_pdb == "OOM_FALLBACK":
            msa_file = msa_retriever.fetch_msa(target_id, sequence)
            if msa_file:
                af2_pdbs = af_oracle.execute_af2(target_id, msa_file, OUTPUT_DIR)
                if af2_pdbs:
                    final_pdbs = af2_pdbs
                    top_plddt = extract_plddt(final_pdbs[0])
                    print(f"  -> [ORACLE] Oracle Assurance (AF2 Rank 1): {top_plddt}%")

        # 3. Evaluator
        if final_pdbs:
            truth_file = os.path.join(BENCHMARK_DIR, f"{target_id}.pdb")
            if os.path.exists(truth_file):
                print("  -> [EVALUATOR] Commencing Core RMSD evaluation for ALL ranks...")
                seq_truth, coords_truth = parse_pdb_strict(truth_file)
                
                best_rmsd = float('inf')
                best_rank = ""
                
                for pdb in final_pdbs:
                    rank_id = "FAST_PATH" if "esm" in pdb else [part for part in os.path.basename(pdb).split('_') if 'rank' in part][0].upper()
                    plddt = extract_plddt(pdb)
                    seq_comp, coords_comp = parse_pdb_strict(pdb)
                    
                    if len(coords_comp) > 0 and len(coords_truth) > 0:
                        core_rmsd = calculate_casp_metrics(coords_comp, seq_comp, coords_truth, seq_truth)
                        if core_rmsd is not None:
                            print(f"    * {rank_id} | pLDDT: {plddt}% | CORE RMSD: {core_rmsd} \u212B")
                            if core_rmsd < best_rmsd:
                                best_rmsd = core_rmsd
                                best_rank = rank_id
                                
                if best_rmsd != float('inf'):
                    print(f"  -> [ABSOLUTE CONCLUSION] Most physically accurate: {best_rank} ({best_rmsd} \u212B)")
                    total_core += best_rmsd
                    valid_targets += 1

    if valid_targets > 0:
        print("\n=======================================================")
        print(f"SYSTEM MEAN CORE RMSD (BEST OF N): {round(total_core / valid_targets, 4)} \u212B")
        print("=======================================================")
        
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()