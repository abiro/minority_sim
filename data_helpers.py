import yaml
import pandas as pd


def load_configs(simulation_paths):
    sim_configs = []
    for sp in simulation_paths:
        with open(sp / 'config.yaml') as f:
            sc = yaml.load(f, Loader=yaml.SafeLoader)
            sim_configs.append(sc)
    return sim_configs


def get_config_table(simulation_paths):
    configs = load_configs(simulation_paths)
    table = [['sim_name']]
    for sp in simulation_paths:
        table[0].append(sp.name)
    for k in sorted(configs[0].keys()):
        table.append([k])
        for conf in configs:
            table[-1].append(conf[k])
    return table


def get_aggregate_df(sim_dir):
    aggregate = {
        'steps': [],
        'p_inflex': [],
        'p_app': []
    }
    for p in sim_dir.glob('*.csv'):
        df = pd.read_csv(p)
        aggregate['steps'].append(df['t'].max())
        aggregate['p_inflex'].append(df['p_inflex'].max())
        aggregate['p_app'].append(df['p_app'].max())
    return pd.DataFrame(aggregate)


def get_simulation_df(sim_dir, sim_id):
    fn = sim_id + '.csv'
    return pd.read_csv(sim_dir / fn)
