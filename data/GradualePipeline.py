import os
import numpy as np
import pandas as pd


class GradualePipeline:
    def __init__(self, alphabet=None):
        self.alphabet = alphabet or list("abcdefghjklmnopqrs")

        self.variant_map = {
            "Pa776": "west", "Pa903": "west", "Mop159": "west",
            "Zü106": "east", "Klo407": "east",
            "Bv34": "italy", "Mod7": "italy"
        }
        self.region_categories = ["west", "east", "italy"]

    def _number_sequence(self, sequence):
        return [self.alphabet.index(pitch) for pitch in sequence if pitch in self.alphabet]

    def _build_tensors(self, sequences):
        lengths = [len(seq) - 1 for seq in sequences]
        max_len = max(lengths)
        num_seqs = len(sequences)

        x_0_data = np.array([seq[0] for seq in sequences], dtype=np.int64)
        x_t_data = np.full((max_len, num_seqs), -1, dtype=np.int64)

        for i, seq in enumerate(sequences):
            seq_len = lengths[i]
            x_t_data[:seq_len, i] = seq[1:]

        return x_0_data, x_t_data, np.array(lengths, dtype=np.int64)

    def process_dataset(self, df, include_mode=False, include_region=False):
        df = df.dropna(subset=["volpiano_pitches"]).copy()
        df["clean_seq"] = df["volpiano_pitches"].apply(self._number_sequence)
        df = df[df["clean_seq"].map(len) > 1].copy()

        if include_mode:
            df = df.dropna(subset=["modus"])
            df = df[df["modus"].astype(str).isin(["1", "2", "3", "4", "5", "6", "7", "8"])]
            df["m_obs"] = df["modus"].astype(int) - 1

        if include_region:
            df = df.dropna(subset=["variant"])
            df["region_name"] = df["variant"].map(self.variant_map)
            df = df.dropna(subset=["region_name"])
            df["region_obs"] = pd.Categorical(df["region_name"], categories=self.region_categories).codes

        valid_sequences = df["clean_seq"].tolist()
        x_0_data, x_t_data, lengths_data = self._build_tensors(valid_sequences)

        out_dict = {
            "x_0_obs": x_0_data,
            "x_t_obs": list(x_t_data.T),
            "lengths_obs": lengths_data
        }

        if include_mode:
            out_dict["m_obs"] = df["m_obs"].to_numpy(dtype=np.int64)
            out_dict["num_modes"] = 8

        if include_region:
            out_dict["region_obs"] = df["region_obs"].to_numpy(dtype=np.int64)
            out_dict["num_regions"] = len(self.region_categories)

        return pd.DataFrame(out_dict)

    def generate_all_files(self, input_csv, output_dir="data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        raw_df = pd.read_csv(input_csv)

        df_pitches = self.process_dataset(raw_df, include_mode=False, include_region=False)
        p1 = os.path.join(output_dir, "dataset_pitches.pkl")
        df_pitches.to_pickle(p1)
        print(f"Saved {p1} (N={len(df_pitches)})")

        df_mode = self.process_dataset(raw_df, include_mode=True, include_region=False)
        p2 = os.path.join(output_dir, "dataset_pitches_modes.pkl")
        df_mode.to_pickle(p2)
        print(f"Saved {p2} (N={len(df_mode)})")

        df_region = self.process_dataset(raw_df, include_mode=True, include_region=True)
        p3 = os.path.join(output_dir, "dataset_pitches_modes_regions.pkl")
        df_region.to_pickle(p3)
        print(f"Saved {p3} (N={len(df_region)})")

    @staticmethod
    def load(filepath):
        df = pd.read_pickle(filepath)

        data_dict = {
            "x_0_obs": df["x_0_obs"].to_numpy(dtype=np.int64),
            "x_t_obs": np.stack(df["x_t_obs"].values).T,
            "lengths_obs": df["lengths_obs"].to_numpy(dtype=np.int64),
        }

        if "m_obs" in df.columns:
            data_dict["m_obs"] = df["m_obs"].to_numpy(dtype=np.int64)
            data_dict["num_modes"] = int(df["num_modes"].iloc[0])

        if "region_obs" in df.columns:
            data_dict["region_obs"] = df["region_obs"].to_numpy(dtype=np.int64)
            data_dict["num_regions"] = int(df["num_regions"].iloc[0])

        return data_dict