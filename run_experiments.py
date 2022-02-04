from data import load_adult, preprocess_adult
from metrics import eval_model
from train import train_decaf, train_fairgan, train_vanilla_gan, train_wgan_gp

# Define DAG for Adult dataset
DAG = [
    # Edges from race
    ['race', 'occupation'],
    ['race', 'income'],
    ['race', 'hours-per-week'],
    ['race', 'education'],
    ['race', 'marital-status'],

    # Edges from age
    ['age', 'occupation'],
    ['age', 'hours-per-week'],
    ['age', 'income'],
    ['age', 'workclass'],
    ['age', 'marital-status'],
    ['age', 'education'],
    ['age', 'relationship'],
    
    # Edges from sex
    ['sex', 'occupation'],
    ['sex', 'marital-status'],
    ['sex', 'income'],
    ['sex', 'workclass'],
    ['sex', 'education'],
    ['sex', 'relationship'],
    
    # Edges from native country
    ['native-country', 'marital-status'],
    ['native-country', 'hours-per-week'],
    ['native-country', 'education'],
    ['native-country', 'workclass'],
    ['native-country', 'income'],
    ['native-country', 'relationship'],
    
    # Edges from marital status
    ['marital-status', 'occupation'],
    ['marital-status', 'hours-per-week'],
    ['marital-status', 'income'],
    ['marital-status', 'workclass'],
    ['marital-status', 'relationship'],
    ['marital-status', 'education'],
    
    # Edges from education
    ['education', 'occupation'],
    ['education', 'hours-per-week'],
    ['education', 'income'],
    ['education', 'workclass'],
    ['education', 'relationship'],
    
    # All remaining edges
    ['occupation', 'income'],
    ['hours-per-week', 'income'],
    ['workclass', 'income'],
    ['relationship', 'income'],
]


def dag_to_idx(df, dag):
    """Convert columns in a DAG to the corresponding indices."""

    dag_idx = []
    for edge in dag:
        dag_idx.append([df.columns.get_loc(edge[0]), df.columns.get_loc(edge[1])])

    return dag_idx


def create_bias_dict(df, edge_map):
    """
    Convert the given edge tuples to a bias dict used for generating
    debiased synthetic data.
    """
    bias_dict = {}
    for key, val in edge_map.items():
        bias_dict[df.columns.get_loc(key)] = [df.columns.get_loc(f) for f in val]
    
    return bias_dict


def train_models(num_runs=10):
    dataset_train = preprocess_adult(load_adult())
    dataset_test = preprocess_adult(load_adult(test=True))

    print('Size of dataset:', len(dataset_train), len(dataset_test))

    dag_seed = dag_to_idx(dataset_train, DAG)

    results = {
        'original': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'vanilla_gan': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'wgan_gp': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'fairgan': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'decaf_nd': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'decaf_dp': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'decaf_cf': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
        'decaf_ftu': {'precision': [], 'recall': [], 'auroc': [], 'dp': [], 'ftu': []},
    }

    bias_dict_ftu = create_bias_dict(dataset_train, {'income': ['sex']})
    bias_dict_dp = create_bias_dict(dataset_train, {'income': [
        'occupation', 'hours-per-week', 'marital-status', 'education', 'sex',
        'workclass', 'relationship']})
    bias_dict_cf = create_bias_dict(dataset_train, {'income': [
        'marital-status', 'sex']})
    bias_dicts = {'nd': {}, 'dp': bias_dict_dp, 'cf': bias_dict_cf, 'ftu': bias_dict_ftu}

    for model in ['original', 'vanilla_gan', 'wgan_gp', 'fairgan', 'decaf']:
        for run in range(num_runs):
            train_func = None
            train_kwargs = {}
            if model == 'vanilla_gan':
                train_func  = train_vanilla_gan
            elif model == 'wgan_gp':
                train_func = train_wgan_gp
            elif model == 'fairgan':
                train_func = train_fairgan
            elif model == 'decaf':
                train_func = train_decaf
                train_kwargs['dag_seed'] = dag_seed

            if model == 'original':
                model_results = eval_model(dataset_train, dataset_test)
                for key, value in model_results.items():
                    results[model][key].append(value)
            else:
                if model == 'decaf':
                    for bias_dict in bias_dicts.keys():
                        train_kwargs['biased_edges'] = bias_dicts[bias_dict]
                        synth_data = train_func(
                            dataset_train,
                            model_name=f'{model}_experiment_1_run_{run+1}',
                            **train_kwargs)
                        model_results = eval_model(synth_data, dataset_test)
                        for key, value in model_results.items():
                            results[f'{model}_{bias_dict}'][key].append(value)
                else:
                    synth_data = train_func(dataset_train,
                                            model_name=f'{model}_experiment_1_run_{run+1}',
                                            **train_kwargs)
                    model_results = eval_model(synth_data, dataset_test)
                    for key, value in model_results.items():
                        results[model][key].append(value)

    for model in results.keys():
        print(f'{model}: {results[model]}')


if __name__ == '__main__':
    train_models(num_runs=10)
