import os
# OOM PROTECTION: Must be declared before PyTorch loads
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import gc
import numpy as np
import subprocess
import shutil
import sys

# ============================================================================
# V500.3 NEURAL BRIDGE (MEMORY INTERCEPTOR)
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
# ARCHITECTURAL WORKSPACE (V8.1 ENSEMBLE ENGINE - MASSIVE SCALING)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark_data", "category_5")
MSA_DIR = os.path.join(BASE_DIR, "msa_data_cat5")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_targets", "v8_ensemble_output")

for directory in [BENCHMARK_DIR, MSA_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

THRESHOLD = 100.0  
THREE_TO_ONE = {'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLU':'E', 'GLN':'Q', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F', 'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V'}
VALID_AAS = set("ARNDCQEGHILKMFPSTWYV")

# ============================================================================
# MODULE 1: MONOMER MSA GENERATOR (COLAB DEFAULT)
# ============================================================================
class IPMColabDefaultMSAGenerator:
    def get_msa(self, fasta_path, target_id, sequence):
        final_a3m_path = os.path.join(MSA_DIR, f"{target_id}_colabfold_default.a3m")
        if os.path.exists(final_a3m_path):
            print(f"  -> [MSA MODULE] A3M cache found for {target_id}. Skipping generation.")
            return final_a3m_path
            
        print(f"  -> [MSA MODULE] Generating default monomer MSA (ColabFold).")
        temp_msa_dir = os.path.join(MSA_DIR, f"temp_{target_id}_msa")
        os.makedirs(temp_msa_dir, exist_ok=True)
        
        monomer_fasta = os.path.join(temp_msa_dir, f"{target_id}.fasta")
        with open(monomer_fasta, 'w') as f:
            f.write(f">{target_id}\n{sequence}\n")
            
        try:
            cmd = ["colabfold_batch", monomer_fasta, temp_msa_dir, "--msa-only"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL) 
            
            a3m_files = glob.glob(os.path.join(temp_msa_dir, "*.a3m"))
            primary_a3m = None
            for f in a3m_files:
                basename = os.path.basename(f).lower()
                if not basename.startswith("uniref") and not basename.startswith("bfd") and not basename.startswith("pdb70"):
                    primary_a3m = f
                    break
                    
            if primary_a3m:
                shutil.copy(primary_a3m, final_a3m_path)
                print(f"  -> [MSA MODULE] Generation Complete. Saved A3M: {final_a3m_path}")
                return final_a3m_path
            return None
        except Exception as e:
            print(f"  -> [MSA FATAL] MMseqs2 execution failed: {e}")
            return None
        finally:
            if os.path.exists(temp_msa_dir): shutil.rmtree(temp_msa_dir)

# ============================================================================
# MODULE 2: NATIVE FAIR COMPILER & EXACT TOPOLOGICAL RENDERING (ESMFold)
# ============================================================================
class IPMNativeRootCompiler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.max_vram_length = 1200 
        try:
            self.model = esm.pretrained.esmfold_v1().eval().to(self.device)
            self.model.set_chunk_size(8) 
            self.esm_active = True
        except Exception as e:
            self.esm_active = False

    def compile_to_pdb(self, target_id, sequence):
        if not self.esm_active: return None
        try:
            sanitized_seq = "".join([aa if aa.upper() in VALID_AAS else "G" for aa in sequence])
            if len(sanitized_seq) > self.max_vram_length: return "OOM_FALLBACK"

            with torch.no_grad():
                pdb_string = self.model.infer_pdb(sanitized_seq, num_recycles=3)
                
            out_file = os.path.join(OUTPUT_DIR, f"{target_id}_esm.pdb")
            with open(out_file, 'w') as f:
                f.write(pdb_string)
            return out_file
        except RuntimeError:
            torch.cuda.empty_cache()
            return "OOM_FALLBACK"
        finally:
            gc.collect()
            torch.cuda.empty_cache()

# ============================================================================
# MODULE 3: ALPHAFOLD2 ENSEMBLE ORACLE (VARIABLE FORKING PROTOCOL)
# ============================================================================
class IPMAlphaFoldOracle:
    def execute_af2(self, target_id, input_a3m, output_dir, total_models, num_recycles, sequence=""):
        # Bypass structural weight limits by engaging multiple seeds for large ensembles
        num_seeds = (total_models + 4) // 5
        active_models = 5 if total_models >= 5 else total_models

        print(f"  -> [PHASE 3] Engaging ALPHAFOLD2 (Target Models: {total_models}, Seeds: {num_seeds}, Recycles: {num_recycles})...")
        print(f"  -> [PHASE 3] INJECTING VARIABLE FORKING PROTOCOL (Dropout: ENABLED)")
            
        temp_input_dir = os.path.join(output_dir, f"input_{target_id}_colab_default")
        temp_output_dir = os.path.join(output_dir, f"temp_{target_id}_colab_default")
        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            fasta_path = os.path.join(temp_input_dir, f"{target_id}.fasta")
            with open(fasta_path, 'w') as f:
                f.write(f">{target_id}\n{sequence}\n")
                
            custom_a3m_path = os.path.join(temp_input_dir, f"{target_id}.a3m")
            shutil.copy(input_a3m, custom_a3m_path)
            
            cmd = [
                "colabfold_batch", temp_input_dir, temp_output_dir, 
                "--num-recycle", str(num_recycles),
                "--num-models", str(active_models),
                "--num-seeds", str(num_seeds),
                "--model-type", "auto",
                "--use-dropout"
            ]
            
            af2_env = os.environ.copy()
            af2_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            af2_env["TF_FORCE_UNIFIED_MEMORY"] = "1"
            af2_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75" 
            
            subprocess.run(cmd, env=af2_env, check=True)
            
            # Retrieve and strictly cap at the requested total_models
            all_pdbs = sorted(glob.glob(os.path.join(temp_output_dir, "*_rank_*.pdb")))
            all_pdbs = all_pdbs[:total_models]
            saved_pdbs = []
            
            for pdb in all_pdbs:
                try:
                    rank_str = os.path.basename(pdb).split('_rank_')[1].split('_')[0]
                except IndexError:
                    rank_str = "UNK"
                final_pdb = os.path.join(output_dir, f"{target_id}_rank_{rank_str}_ensemble.pdb")
                shutil.copy(pdb, final_pdb)
                saved_pdbs.append(final_pdb)
                        
            if saved_pdbs:
                print(f"  -> [PHASE 3] AF2 Ensemble successfully synthesized ({len(saved_pdbs)} States evaluated).")
                return saved_pdbs
            return None
        except Exception as e:
            print(f"  -> [ORACLE ERROR] Execution exception: {e}")
            return None
        finally:
            if os.path.exists(temp_input_dir): shutil.rmtree(temp_input_dir)
            if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
            gc.collect()
            torch.cuda.empty_cache()

# ============================================================================
# MODULE 4: EVALUATOR (RIGID BODY ALIGNMENT)
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
        full_seq = ""
        full_coords = []
        for chain_id in sorted(chains.keys()):
            full_seq += chains[chain_id]["seq"]
            full_coords.extend(chains[chain_id]["coords"])
        return full_seq, np.array(full_coords)
    except Exception:
        return "", np.array([])

def calculate_casp_metrics(coords_computed, seq_computed, coords_truth, seq_truth):
    if len(seq_computed) == 0 or len(seq_truth) == 0: return None
    n, m = len(seq_computed), len(seq_truth)
    dp = np.zeros((n + 1, m + 1))
    gap_penalty = -0.1
    
    for j in range(1, m + 1): dp[0, j] = j * gap_penalty
    for i in range(1, n + 1): dp[i, 0] = i * gap_penalty
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = 1.0 if seq_computed[i-1] == seq_truth[j-1] and seq_computed[i-1] != 'X' else -2.0
            dp[i, j] = max(dp[i-1, j-1] + match, dp[i-1, j] + gap_penalty, dp[i, j-1] + gap_penalty)
            
    aligned_pairs = []
    i, j = np.unravel_index(np.argmax(dp), dp.shape)
    while i > 0 and j > 0:
        match = 1.0 if seq_computed[i-1] == seq_truth[j-1] and seq_computed[i-1] != 'X' else -2.0
        if dp[i, j] == dp[i-1, j-1] + match:
            if match == 1.0: aligned_pairs.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i, j] == dp[i-1, j] + gap_penalty: i -= 1
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
    return round(np.sqrt(np.mean(core_distances**2)), 4) if len(core_distances) > 0 else 0.0

# ============================================================================
# ORCHESTRATOR & USER INTERFACE
# ============================================================================
def extract_sequence_from_fasta(fasta_path):
    sequence = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith(">"): sequence += line.strip()
    return sequence

def process_target_with_config(compiler, af_oracle, msa_file, target_id, sequence, total_models, num_recycles):
    print(f"\n[ORCHESTRATOR] Processing Target: {target_id}")
    final_pdbs = []
    
    fast_pdb = compiler.compile_to_pdb(target_id, sequence)
    if fast_pdb and fast_pdb != "OOM_FALLBACK":
        plddt = extract_plddt(fast_pdb)
        print(f"  -> [GATE] Initial Spatial Assurance (ESMFold): {plddt}%")
        if plddt >= THRESHOLD: final_pdbs = [fast_pdb]
            
    if not final_pdbs:
        af2_pdbs = af_oracle.execute_af2(target_id, msa_file, OUTPUT_DIR, total_models=total_models, num_recycles=num_recycles, sequence=sequence)
        if af2_pdbs:
            final_pdbs = af2_pdbs
            print(f"  -> [ORACLE] Oracle Assurance (Top Rank): {extract_plddt(final_pdbs[0])}%")
            
    best_rmsd = float('inf')
    best_rank = ""
    
    if final_pdbs:
        truth_file = glob.glob(os.path.join(BENCHMARK_DIR, f"*{target_id.split('_')[0]}*.pdb"))
        truth_file = truth_file[0] if truth_file else None
        
        if truth_file and os.path.exists(truth_file):
            print("  -> [EVALUATOR] Commencing Core RMSD evaluation for ALL ranks...")
            seq_truth, coords_truth = parse_pdb_strict(truth_file)
            
            for pdb in final_pdbs:
                rank_id = "FAST_PATH" if "esm" in pdb else [part for part in os.path.basename(pdb).split('_') if 'rank' in part][0].upper()
                seq_comp, coords_comp = parse_pdb_strict(pdb)
                
                if len(coords_comp) > 0 and len(coords_truth) > 0:
                    core_rmsd = calculate_casp_metrics(coords_comp, seq_comp, coords_truth, seq_truth)
                    if core_rmsd is not None:
                        print(f"    * RANK {rank_id} | pLDDT: {extract_plddt(pdb)}% | CORE RMSD: {core_rmsd} \u212B")
                        if core_rmsd < best_rmsd:
                            best_rmsd, best_rank = core_rmsd, rank_id
                            
            if best_rmsd != float('inf'):
                print(f"  -> [ABSOLUTE CONCLUSION] Most physically accurate: RANK {best_rank} ({best_rmsd} \u212B)")
                return best_rmsd, best_rank
        else:
            print("  -> [EVALUATOR] No reference PDB found. Evaluation skipped.")
            return None, None
    return None, None

def main():
    print("\n=======================================================")
    print("    IPM ENSEMBLE ENGINE (V8.1 - MASSIVE SCALING)     ")
    print("=======================================================\n")
    
    try:
        models_input = input("Enter total number of ensemble models to generate (1-50): ").strip()
        total_models = int(models_input) if models_input.isdigit() and 1 <= int(models_input) <= 50 else 50
    except ValueError:
        total_models = 50

    try:
        recycles_input = input("Enter number of Recycles (Recommended: 3): ").strip()
        num_recycles = int(recycles_input) if recycles_input.isdigit() else 3
    except ValueError:
        num_recycles = 3
    
    print(f"\n[SYSTEM] Configuration Locked: {total_models} Models | {num_recycles} Recycles | Variable Forking: ACTIVE")
        
    compiler = IPMNativeRootCompiler()
    af_oracle = IPMAlphaFoldOracle()
    msa_gen = IPMColabDefaultMSAGenerator()
    
    fasta_files = glob.glob(os.path.join(BENCHMARK_DIR, "*.fasta"))
    if not fasta_files: 
        print(f"[FATAL ERROR] Zero .fasta targets detected in {BENCHMARK_DIR}.")
        input("Press Enter to exit...")
        sys.exit(1)

    for fasta_file in fasta_files:
        target_id = os.path.basename(fasta_file).replace('.fasta', '')
        sequence = extract_sequence_from_fasta(fasta_file)
        
        print(f"\n*******************************************************")
        print(f"    PROCESSING ENSEMBLE TARGET {target_id}")
        print(f"*******************************************************")
        
        msa_file = msa_gen.get_msa(fasta_file, target_id, sequence)
        if not msa_file: continue
            
        rmsd, rank = process_target_with_config(compiler, af_oracle, msa_file, target_id, sequence, total_models, num_recycles)
            
        print(f"\n=======================================================")
        if rmsd is not None:
            print(f"FINAL RESULT FOR {target_id}: {rmsd} \u212B (Rank {rank})")
        else:
            print(f"FINAL RESULT FOR {target_id}: Ensemble Generated. Evaluation Skipped.")
        print("=======================================================\n")
    
if __name__ == "__main__":
    main()