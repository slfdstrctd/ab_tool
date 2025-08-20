import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise ValueError(f"Data directory {data_directory} does not exist")

        self.users_df = None
        self.messages_df = None
        self.payments_df = None
        self.experiments = {}

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading all data files...")

        self.users_df = self._load_csv_files("users_all_*.csv")
        logger.info(f"Loaded {len(self.users_df)} user records")

        self.messages_df = self._load_csv_files("messages_all_*.csv")
        logger.info(f"Loaded {len(self.messages_df)} message records")

        self.payments_df = self._load_csv_files("payments_all_*.csv")
        logger.info(f"Loaded {len(self.payments_df)} payment records")

        self._parse_experiment_flags()

        return {
            "users": self.users_df,
            "messages": self.messages_df,
            "payments": self.payments_df,
        }

    def _load_csv_files(self, pattern: str) -> pd.DataFrame:
        files = list(self.data_directory.glob(pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {pattern}")

        dfs = []
        for file in sorted(files):
            logger.debug(f"Loading {file}")
            df = pd.read_csv(file)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)

        if "user_id" in combined_df.columns:
            if "ts" in combined_df.columns:
                combined_df['ts'] = pd.to_datetime(combined_df['ts'])
                combined_df = combined_df.sort_values(['user_id', 'ts'])
            combined_df = combined_df.drop_duplicates(subset=["user_id"], keep="last")

        return combined_df

    def _parse_experiment_flags(self):
        logger.info("Parsing experiment flags...")

        def safe_parse_json(json_str):
            try:
                if pd.isna(json_str):
                    return {}

                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
            except:
                logger.warning(f"Failed to parse JSON: {json_str}")
                return {}

        self.users_df["experiments"] = self.users_df["ampl_user_data"].apply(
            safe_parse_json
        )

        all_experiments = set()
        for exp_dict in self.users_df["experiments"]:
            all_experiments.update(exp_dict.keys())

        self.experiments = list(all_experiments)
        logger.info(f"Found {len(self.experiments)} experiments: {self.experiments}")

        for exp_name in self.experiments:
            self.users_df[f"{exp_name}_group"] = self.users_df["experiments"].apply(
                lambda x: self._get_exp_group(x, exp_name)
            )

    def _get_exp_group(self, exp_dict: Dict, exp_name: str) -> Optional[bool]:
        if exp_name not in exp_dict:
            return None

        value = exp_dict[exp_name]
        if value == "1":
            return True
        elif value == "0":
            return False
        else:
            return None

    def get_experiment_data(self, experiment_name: str) -> Dict[str, pd.DataFrame]:
        if self.users_df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        if not experiment_name.startswith("exp_"):
            full_exp_name = f"exp_{experiment_name}"
        else:
            full_exp_name = experiment_name

        exp_column = f"{full_exp_name}_group"
        if exp_column not in self.users_df.columns:
            available = [
                col.replace("_group", "").replace("exp_", "")
                for col in self.users_df.columns
                if col.endswith("_group")
            ]
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available}"
            )

        exp_users = self.users_df[self.users_df[exp_column].notna()].copy()

        control_users = exp_users[exp_users[exp_column] == False]["user_id"].values
        treatment_users = exp_users[exp_users[exp_column] == True]["user_id"].values

        logger.info(
            f"Experiment '{experiment_name}': "
            f"{len(control_users)} control users, "
            f"{len(treatment_users)} treatment users"
        )

        control_data = self._merge_user_data(control_users)
        treatment_data = self._merge_user_data(treatment_users)

        return {
            "control": control_data,
            "treatment": treatment_data,
            "experiment_name": experiment_name,
            "control_users": control_users,
            "treatment_users": treatment_users,
        }

    def _merge_user_data(self, user_ids: np.ndarray) -> pd.DataFrame:
        users_subset = self.users_df[self.users_df["user_id"].isin(user_ids)].copy()

        if self.messages_df is not None:
            messages_agg = self.messages_df[self.messages_df["user_id"].isin(user_ids)]
            messages_agg = (
                messages_agg.groupby("user_id")["messages_count"].sum().reset_index()
            )
            users_subset = users_subset.merge(messages_agg, on="user_id", how="left")
            users_subset["messages_count"] = users_subset["messages_count"].fillna(0)
        else:
            users_subset["messages_count"] = 0

        if self.payments_df is not None:
            payments_agg = self.payments_df[self.payments_df["user_id"].isin(user_ids)]
            payments_agg = (
                payments_agg.groupby("user_id")["price_usd"].sum().reset_index()
            )
            payments_agg.rename(columns={"price_usd": "revenue_usd"}, inplace=True)
            users_subset = users_subset.merge(payments_agg, on="user_id", how="left")
            users_subset["revenue_usd"] = users_subset["revenue_usd"].fillna(0)
        else:
            users_subset["revenue_usd"] = 0

        return users_subset

    def get_data_quality_report(self) -> Dict[str, Any]:
        report = {
            "total_users": len(self.users_df) if self.users_df is not None else 0,
            "total_messages": len(self.messages_df)
            if self.messages_df is not None
            else 0,
            "total_payments": len(self.payments_df)
            if self.payments_df is not None
            else 0,
            "experiments": self.experiments,
            "experiment_details": {},
        }

        for exp in self.experiments:
            exp_column = f"{exp}_group"
            if exp_column in self.users_df.columns:
                exp_data = self.users_df[exp_column].value_counts(dropna=False)
                report["experiment_details"][exp] = {
                    "control": int(exp_data.get(False, 0)),
                    "treatment": int(exp_data.get(True, 0)),
                    "not_in_experiment": int(
                        exp_data.get(np.nan, 0) if np.nan in exp_data else 0
                    ),
                }

        return report

    def get_all_experiments(self) -> List[str]:
        return self.experiments.copy()
