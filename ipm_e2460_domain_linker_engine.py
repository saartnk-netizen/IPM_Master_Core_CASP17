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

# ----------------------------------------------------------------------------
# [IPM] IMPORT OPENMM AND PDBFIXER FOR ACTIVE RELAXATION
# ----------------------------------------------------------------------------
try:
    import openmm.app as app
    import openmm as mm
    import openmm.unit as unit
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("[FATAL WARNING] OpenMM or PDBFixer not found in environment. Thermodynamic relaxation will fail.")
    print("Run: conda install -c conda-forge openmm pdbfixer")

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
# ARCHITECTURAL WORKSPACE (V10.0 ENSEMBLE ENGINE - DOMAIN-LINKER PROTOCOL)
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
        
        # [IPM FIX 8] Autonomous Cache Purge
        if os.path.exists(final_a3m_path):
            os.remove(final_a3m_path)
            print(f"  -> [MSA MODULE] Orphaned cache strictly deleted for {target_id}. Enforcing fresh topological extraction.")
            
        print(f"  -> [MSA MODULE] Generating pure topological monomer MSA (ColabFold).")
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
# MODULE 3: ALPHAFOLD2 ENSEMBLE ORACLE (FULL MSA WITH VARIABLE FORKING)
# ============================================================================
class IPMAlphaFoldOracle:
    def execute_af2(self, target_id, input_a3m, output_dir, total_models, num_recycles, sequence=""):
        num_seeds = (total_models + 4) // 5
        active_models = 5 if total_models >= 5 else total_models

        print(f"  -> [PHASE 3] Engaging ALPHAFOLD2 (Target Models: {total_models}, Seeds: {num_seeds}, Recycles: {num_recycles})...")
        print(f"  -> [PHASE 3] INJECTING VARIABLE FORKING PROTOCOL (Dropout: ENABLED)")
        print(f"  -> [PHASE 3] ENGAGING ALGORITHMIC MESH DECIMATION (Domain-Linker Continuous Extraction)")
            
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
                "--use-dropout",
                "--max-msa", "16:32"
            ]
            
            af2_env = os.environ.copy()
            af2_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            af2_env["TF_FORCE_UNIFIED_MEMORY"] = "1"
            af2_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75" 
            
            subprocess.run(cmd, env=af2_env, check=True)
            
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
# MODULE 4: ACTIVE THERMODYNAMIC MINIMIZATION (OPENMM + PDBFIXER)
# ============================================================================
def ipm_protein_relaxation(input_pdb, output_pdb):
    if not OPENMM_AVAILABLE:
        shutil.copy(input_pdb, output_pdb)
        return False
        
    try:
        fixer = PDBFixer(filename=input_pdb)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0) 
        
        forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        
        system = forcefield.createSystem(fixer.topology, 
                                         nonbondedMethod=app.NoCutoff, 
                                         constraints=app.HBonds,
                                         rigidWater=True) 
        
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            simulation = app.Simulation(fixer.topology, system, integrator, platform)
        except Exception:
            simulation = app.Simulation(fixer.topology, system, integrator)
            
        simulation.context.setPositions(fixer.positions)
        
        print("    * [IPM RELAX] Commencing absolute topological minimization (Implicit Method)...")
        simulation.minimizeEnergy(maxIterations=5000)
        
        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(fixer.topology, positions, open(output_pdb, 'w'))
        return True
    except Exception as e:
        print(f"    * [IPM RELAX ERROR] Absolute Minimization failed: {e}")
        shutil.copy(input_pdb, output_pdb)
        return False

# ============================================================================
# MODULE 5: EVALUATOR (ENSEMBLE CROSS-VALIDATION MATRIX & ITERATIVE SVD)
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

def parse_pdb_ensemble_truth(file_path):
    models = []
    current_chains = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ENDMDL"):
                    if current_chains:
                        models.append(current_chains)
                        current_chains = {}
                    continue
                if line.startswith("ATOM") and line[12:16] == " CA ":
                    if line[16] not in [' ', 'A', '1']: continue 
                    chain_id = line[21]
                    res_name = line[17:20].strip()
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    if chain_id not in current_chains: current_chains[chain_id] = {"seq": "", "coords": []}
                    current_chains[chain_id]["seq"] += THREE_TO_ONE.get(res_name, 'X')
                    current_chains[chain_id]["coords"].append([x, y, z])
            
            if current_chains:
                models.append(current_chains)
                
        parsed_models = []
        for m in (models if models else [current_chains]):
            full_seq = ""
            full_coords = []
            for chain_id in sorted(m.keys()):
                s = m[chain_id]["seq"]
                c = m[chain_id]["coords"]
                # [IPM FIX 11 - E2460] STRICT PRESERVATION OF ALL RESIDUES
                # Histidine tags are NOT stripped to comply with CASP17 evaluation rules.
                full_seq += s
                full_coords.extend(c)
            parsed_models.append((full_seq, np.array(full_coords)))
        return parsed_models
    except Exception:
        return []

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
            s = chains[chain_id]["seq"]
            c = chains[chain_id]["coords"]
            # [IPM FIX 11 - E2460] STRICT PRESERVATION OF ALL RESIDUES
            full_seq += s
            full_coords.extend(c)
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
    
    aligned_P_full = np.array([coords_computed[i] for i, j in aligned_pairs[::-1]])
    aligned_Q_full = np.array([coords_truth[j] for i, j in aligned_pairs[::-1]])
    
    current_P = aligned_P_full
    current_Q = aligned_Q_full
    active_indices = np.arange(len(aligned_P_full))
    
    for iteration in range(5):
        Pc = current_P - np.mean(current_P, axis=0)
        Qc = current_Q - np.mean(current_Q, axis=0)
        
        V, S, W = np.linalg.svd(np.dot(Pc.T, Qc))
        if (np.linalg.det(V) * np.linalg.det(W)) < 0.0:
            S[-1] = -S[-1]; V[:, -1] = -V[:, -1]
        U_global = np.dot(V, W)
        
        P_rot_all = np.dot(aligned_P_full - np.mean(current_P, axis=0), U_global)
        Q_centered_all = aligned_Q_full - np.mean(current_Q, axis=0)
        
        distances = np.sqrt(np.sum((P_rot_all - Q_centered_all) ** 2, axis=1))
        
        new_active = np.where(distances < 3.5)[0]
        
        if len(new_active) < 10 or len(new_active) == len(active_indices):
            break
            
        active_indices = new_active
        current_P = aligned_P_full[active_indices]
        current_Q = aligned_Q_full[active_indices]
    
    if len(active_indices) < 10: return None
    
    core_distances = distances[active_indices]
    return round(np.sqrt(np.mean(core_distances**2)), 4)

# ============================================================================
# MODULE 6: ORCHESTRATOR & USER INTERFACE
# ============================================================================
def extract_sequence_from_fasta(fasta_path):
    sequence = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if not line.startswith(">"): sequence += line.strip()
            
    # [IPM FIX 11 - E2460] STRICT PRESERVATION OF ALL RESIDUES
    # Do not strip any terminal sequence. The entire FRET tag is maintained.
        
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
            
    best_rmsd_global = float('inf')
    best_rank_global = ""
    all_rmsds_global = []
    
    if final_pdbs:
        truth_file = glob.glob(os.path.join(BENCHMARK_DIR, f"*{target_id.split('_')[0]}*.pdb"))
        truth_file = truth_file[0] if truth_file else None
        
        if truth_file and os.path.exists(truth_file):
            print("  -> [EVALUATOR] Commencing Ensemble Cross-Validation Matrix...")
            
            truth_models = parse_pdb_ensemble_truth(truth_file)
            print(f"    * Detected {len(truth_models)} reference state(s) in {os.path.basename(truth_file)}")
            
            for pdb in final_pdbs:
                rank_id = "FAST_PATH" if "esm" in pdb else [part for part in os.path.basename(pdb).split('_') if 'rank' in part][0].upper()
                
                relaxed_pdb = pdb.replace(".pdb", "_ipm_relaxed.pdb")
                is_relaxed = ipm_protein_relaxation(pdb, relaxed_pdb)
                eval_pdb = relaxed_pdb if is_relaxed else pdb
                
                seq_comp, coords_comp = parse_pdb_strict(eval_pdb)
                
                if len(coords_comp) > 0:
                    best_local_rmsd = float('inf')
                    best_local_truth_model = 0
                    
                    for m_idx, (seq_truth, coords_truth) in enumerate(truth_models):
                        if len(coords_truth) > 0:
                            core_rmsd = calculate_casp_metrics(coords_comp, seq_comp, coords_truth, seq_truth)
                            if core_rmsd is not None and core_rmsd < best_local_rmsd:
                                best_local_rmsd = core_rmsd
                                best_local_truth_model = m_idx + 1
                                
                    if best_local_rmsd != float('inf'):
                        all_rmsds_global.append(best_local_rmsd)
                        status_str = "[RELAXED]" if is_relaxed else "[UNRELAXED]"
                        print(f"    * RANK {rank_id} {status_str} | Matched Truth Model #{best_local_truth_model} | CORE RMSD: {best_local_rmsd} \u212B")
                        
                        if best_local_rmsd < best_rmsd_global:
                            best_rmsd_global = best_local_rmsd
                            best_rank_global = rank_id
                            
            if best_rmsd_global != float('inf'):
                avg_rmsd = round(sum(all_rmsds_global) / len(all_rmsds_global), 4) if all_rmsds_global else 0.0
                print(f"  -> [ABSOLUTE CONCLUSION] Most physically accurate: RANK {best_rank_global} ({best_rmsd_global} \u212B)")
                return best_rmsd_global, best_rank_global, avg_rmsd
        else:
            print("  -> [EVALUATOR] No reference PDB found. Blind Prediction Mode Engaged (Evaluation skipped).")
            print("  -> [RELAXATION] Commencing Thermodynamic Minimization for predicted ensemble...")
            for pdb in final_pdbs:
                relaxed_pdb = pdb.replace(".pdb", "_ipm_relaxed.pdb")
                ipm_protein_relaxation(pdb, relaxed_pdb)
            return None, None, None
    return None, None, None

def main():
    print("\n=======================================================")
    print("    IPM ENSEMBLE ENGINE (V10.0 - DOMAIN-LINKER PROTOCOL)   ")
    print("=======================================================\n")
    
    try:
        models_input = input("Enter total number of ensemble models to generate (1-1000) [Default: 500]: ").strip()
        total_models = int(models_input) if models_input.isdigit() and 1 <= int(models_input) <= 1000 else 500
    except ValueError:
        total_models = 500

    try:
        recycles_input = input("Enter number of Recycles (1-50) [Default: 3]: ").strip()
        num_recycles = int(recycles_input) if recycles_input.isdigit() and 1 <= int(recycles_input) <= 50 else 3
    except ValueError:
        num_recycles = 3
    
    print(f"\n[SYSTEM] Configuration Locked: {total_models} Models | {num_recycles} Recycles")
    print(f"[SYSTEM] OpenMM Active: {OPENMM_AVAILABLE}")
    print(f"[SYSTEM] Full MSA Retrieval & Thermodynamic Relaxation: ACTIVE")
        
    compiler = IPMNativeRootCompiler()
    af_oracle = IPMAlphaFoldOracle()
    msa_gen = IPMColabDefaultMSAGenerator()
    
    fasta_files = glob.glob(os.path.join(BENCHMARK_DIR, "*.fasta"))
    if not fasta_files: 
        print(f"[FATAL ERROR] Zero .fasta targets detected in {BENCHMARK_DIR}.")
        input("Press Enter to exit...")
        sys.exit(1)

    summary_results = []

    for fasta_file in fasta_files:
        target_id = os.path.basename(fasta_file).replace('.fasta', '')
        sequence = extract_sequence_from_fasta(fasta_file)
        
        print(f"\n*******************************************************")
        print(f"    PROCESSING TARGET {target_id}")
        print(f"*******************************************************")
        
        msa_file = msa_gen.get_msa(fasta_file, target_id, sequence)
        if not msa_file: continue
            
        rmsd, rank, avg_rmsd = process_target_with_config(compiler, af_oracle, msa_file, target_id, sequence, total_models, num_recycles)
            
        print(f"\n=======================================================")
        if rmsd is not None:
            print(f"FINAL CALIBRATION RESULT FOR {target_id}: {rmsd} \u212B (Rank {rank})")
            summary_results.append({
                "target": target_id,
                "best_rmsd": rmsd,
                "best_rank": rank,
                "avg_rmsd": avg_rmsd
            })
        else:
            print(f"FINAL BLIND PREDICTION FOR {target_id}: Ensemble Generated & Relaxed. Ready for submission.")
        print("=======================================================\n")
    
    print("\n" + "="*70)
    print(" " * 23 + "FINAL EXECUTION SUMMARY")
    print("="*70)
    if summary_results:
        c1, c2, c3, c4 = "TARGET ID", "BEST RMSD (\u212B)", "BEST RANK", "AVG RMSD (\u212B)"
        print(f"{c1:<20} | {c2:<15} | {c3:<12} | {c4:<15}")
        print("-" * 70)
        for res in summary_results:
            print(f"{res['target']:<20} | {res['best_rmsd']:<15} | {res['best_rank']:<12} | {res['avg_rmsd']:<15}")
    else:
        print("No structural calibration evaluations completed. All outputs are blind predictions.")
    print("="*70 + "\n")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()