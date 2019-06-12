num_trials = 4
variances = [0,0.5,10]
means = [0]
decision_epses = [1,0.5]
hardnesses = [False]
methods = ["PER", "average_over_batch"]

output_dir = "scripts/"



preliminary_string = "#!/bin/bash\n" + "#\n" + "#SBATCH --job-name=test\n" + "#SBATCH --partition=deep\n" + "#SBATCH --time=20:00\n" + "#SBATCH --ntasks=1\n"
"#SBATCH --cpus-per-task=1\n" + "#SBATCH --mem-per-cpu=2G\n\n" 
preliminary_string += "source /sailhome/behzad/cs379c-project/gym/roper/bin/activate\n"

print(preliminary_string)
meta_script = open(output_dir + "meta.sh", "w")

for variance in variances:
    for mean in means:
        for method in methods:
            for decision_eps in decision_epses:
                for hardness in hardnesses:
                    output_txt = open(output_dir + "experiments" + str(method) + "_" 
                                      + str(mean) + "_" + str(variance) + "_" 
                                      + str(decision_eps) + "_" 
                                      + str(hardness) + ".sh", "w")
                    output_txt.write(preliminary_string)
                    output_txt.write("python train.py" + " --num_trials " + str(num_trials) 
                                     + " --method " + method + " --variance " + str(variance) 
                                     + " --mean " + str(mean) + " --decision_eps " + str(decision_eps)
                                     + " --hardcoded " + str(hardness) + "\n")
                    meta_script.write("sbatch " + output_dir + "experiments" + str(method) + "_" 
                                      + str(mean) + "_" + str(variance) + "_" 
                                      + str(decision_eps) + "_" 
                                      + str(hardness) + ".sh\n")
                    output_txt.close()
                    
meta_script.close()
