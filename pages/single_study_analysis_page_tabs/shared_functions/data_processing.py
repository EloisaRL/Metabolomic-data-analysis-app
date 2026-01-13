import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def da_testing(self):
        '''
        Performs differential analysis testing, adds pval_df attribute containing results.
        '''
        if self.pathway_level == True:
            dat = self.pathway_data
        else:
            dat = self.processed_data

        # Directly filter using the 'group_type' column
        X_case = dat.loc[dat['group_type'] == 'Case'].select_dtypes(include='number')
        X_ctrl = dat.loc[dat['group_type'] == 'Control'].select_dtypes(include='number')

        
        # Convert to DataFrame if filtering returned a Series
        if isinstance(X_case, pd.Series):
            X_case = X_case.to_frame().T
        if isinstance(X_ctrl, pd.Series):
            X_ctrl = X_ctrl.to_frame().T
        
        # Restrict to common numeric columns
        common_cols = X_case.columns.intersection(X_ctrl.columns)
        X_case = X_case[common_cols]
        X_ctrl = X_ctrl[common_cols]

        stat, pvals = stats.ttest_ind(X_case, X_ctrl,
                                    alternative='two-sided',
                                    nan_policy='raise')
        pval_df = pd.DataFrame({
            'P-value': pvals,
            'Stat': stat,
            'Direction': ['Up' if s > 0 else 'Down' for s in stat]
        }, index=X_case.columns)
        
        pval_df['Stat'] = stat
        pval_df['Direction'] = ['Up' if x > 0 else 'Down' for x in stat]
        self.pval_df = pval_df

        # fdr correction 
        pval_df['FDR_P-value'] = multipletests(pvals, method='fdr_bh')[1]

        # return significant metabolites
        self.DA_metabolites = pval_df[pval_df['FDR_P-value'] < 0.05].index.tolist()
        print(f"Number of differentially abundant metabolites: {len(self.DA_metabolites)}") 

        # generate tuples for nx links
        self.connection = [(self.node_name, met) for met in self.DA_metabolites]
        self.full_connection = [(self.node_name, met) for met in self.processed_data.columns[:-1]]
