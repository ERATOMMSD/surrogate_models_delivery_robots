import os
import pandas as pd


def load_experiment(base_dir, objectives_labels, variables_labels):
    individuals = pd.DataFrame()
    rec_individuals = pd.DataFrame()
    result = pd.DataFrame()
    time_df = pd.DataFrame()
    sim_directories = sorted(
        [d for d in list(next(os.walk(base_dir))[1]) if d[0] != '.'])

    for sim_dir in sim_directories:
        algo_directories = next(os.walk(os.path.join(base_dir, sim_dir)))[1]
        for algo_dir in algo_directories:
            problem_directories = next(
                os.walk(os.path.join(base_dir, sim_dir, algo_dir)))[1]
            for problem_dir in problem_directories:
                gen_directories = next(
                    os.walk(os.path.join(base_dir, sim_dir, algo_dir, problem_dir)))[1]
                for gen_dir in gen_directories:
                    [generations, _] = gen_dir.split('_')
                    req_directories = next(
                        os.walk(os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir)))[1]
                    for req_dir in req_directories:
                        [_, min_req, max_req] = req_dir.split('_')
                        files = next(os.walk(
                            os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir)))[2]
                        var_files = {}
                        fun_files = {}
                        for file in files:
                            repetition = file.split('.')[1]
                            prefix = file.split('.')[0]
                            if prefix == "IND":
                                experiment = pd.read_csv(
                                    os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir, file))
                                experiment['run'] = int(repetition)
                                experiment['problem'] = problem_dir
                                experiment['approach'] = sim_dir
                                experiment['generations'] = generations
                                experiment['min_req'] = min_req
                                experiment['max_req'] = max_req
                                individuals = pd.concat(
                                    (individuals, experiment), ignore_index=True)
                            elif prefix == "REC_IND":
                                experiment = pd.read_csv(
                                    os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir, file))
                                experiment['run'] = int(repetition)
                                experiment['problem'] = problem_dir
                                experiment['approach'] = sim_dir
                                experiment['generations'] = generations
                                experiment['min_req'] = min_req
                                experiment['max_req'] = max_req
                                rec_individuals = pd.concat(
                                    (rec_individuals, experiment), ignore_index=True)
                            elif prefix == "VAR":
                                var_df = pd.read_csv(
                                    os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir, file), delimiter=' ', header=None)
                                # There is a blank column at the end
                                last_column = len(variables_labels)
                                var_df = var_df.drop(columns=[last_column])
                                var_df.columns = variables_labels
                                var_files[repetition] = var_df
                            elif prefix == "FUN":
                                fun_df = pd.read_csv(
                                    os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir, file), delimiter=' ', header=None)
                                # There is a blank column at the end
                                last_column = len(objectives_labels)
                                fun_df = fun_df.drop(columns=[last_column])
                                fun_df.columns = objectives_labels
                                fun_df['run'] = int(repetition)
                                fun_df['problem'] = problem_dir
                                fun_df['approach'] = sim_dir
                                fun_df['generations'] = generations
                                fun_df['min_req'] = min_req
                                fun_df['max_req'] = max_req
                                fun_files[repetition] = fun_df
                            elif prefix == "TIME":
                                file1 = open(
                                    os.path.join(base_dir, sim_dir, algo_dir, problem_dir, gen_dir, req_dir, file), 'r')
                                print(file1)
                                total_time = float(str(file1.readlines()[0]))
                                time_df = pd.concat((time_df, pd.DataFrame({'run': [repetition], 'problem': [problem_dir], 'approach': [sim_dir], 'time': [total_time],
                                                                            'generations': [generations],
                                                                            'min_req': [min_req], 'max_req': [max_req]})), ignore_index=True)

                        for repetition, var_df in var_files.items():
                            fun_df = fun_files[repetition]
                            combined_df = var_df.join(fun_df)
                            result = pd.concat(
                                (result, combined_df), ignore_index=True)

    # Renaming the original id of each individual dataframe
    individuals = individuals.rename(columns={'Unnamed: 0': 'individual_id'})
    result = result.rename(columns={'Unnamed: 0': 'idx'})

    return individuals, rec_individuals, result, time_df


def get_experiment_run(df, problem, approach, run):
    return df[(df.problem == problem) & (df.approach == approach) & (df.run == run)]
