import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

# Visualizations for Final Notebook

# Create a list with the top 10 crimes
top_10_crimes = ['DWI', 'ASSAULT WITH INJURY', 'THEFT BY SHOPLIFTING', 'HARASSMENT', 
                 'AUTO THEFT', 'ASSAULT W/INJURY-FAM/DATE VIOL', 'CRIMINAL MISCHIEF', 
                 'FAMILY DISTURBANCE', 'THEFT', 'BURGLARY OF VEHICLE']

# Create a subsetted df that only uses the top 10 crimes
def subset_top_crimes(train):
    top_crimes_df = train.copy()
    top_crimes_df = top_crimes_df[top_crimes_df.crime_type.isin(top_10_crimes)]
    return top_crimes_df

def plot_cleared(df):
    plt.title("Distribution of Clearance", fontsize=15)
    sns.countplot(y="cleared", data=df)
    plt.ylabel('Cleared', fontsize=12)
    plt.xlabel('Clearance Rate', fontsize=12)
    plt.show()
    
# Create subsetted dfs with caseloads for all districts, the highest caseload district and the lowest caseload district

def subset_districts(train):
    train2 = train.copy()
    train2['counts'] = train2.groupby(['council_district'])['crime_type'].transform('count')
    overall_sample = train2.groupby('council_district').council_district.count().mean()
    nine_sample = train2[train2.council_district == 9].counts
    eight_sample = train2[train2.council_district == 8].counts
    
    return overall_sample, nine_sample, eight_sample
    
# Create a data frame for time-series analysis

def time_series_df(train):
    train2 = train.copy()
    train2 = train2.set_index('occurrence_date').sort_index()
    #Split by month first
    train2['month'] = train2.index.month_name()
    #Split by weekdays
    train2['weekdays'] = train2.index.day_name()
    #Split by year
    train2['year'] = train2.index.year
    # Order the weekdays correctly
    train2['weekdays'] = pd.Categorical(train2['weekdays'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                                        'Friday', 'Saturday', 'Sunday'])
    # Order the months correctly
    train2['month'] = pd.Categorical(train2['month'], categories=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                                            'September','October', 'November', 'December'])
    return train2

# Create subsets of data only including Friday or not including Friday for an independent t-test
def friday_subsets(time_series_df):
    friday_only = ['Friday']
    subset_friday = time_series_df.copy()
    subset_friday = subset_friday[subset_friday.weekdays.isin(friday_only)]
    subset_not_friday = time_series_df.copy()
    subset_not_friday = subset_not_friday[~subset_not_friday.weekdays.isin(friday_only)]#Without friday
    
    return subset_friday, subset_not_friday

# Create a dataframe that is prepare for time-series analysis on crime reporting time

def report_time_df(train):
    # Calculate a time_to_report feature
    report_time_df = train.copy()
    report_time_df['time_to_report'] = report_time_df.report_time - report_time_df.occurrence_time
    report_time_df['time_to_report_bins'] = pd.cut(
    report_time_df.time_to_report,
    [
        pd.Timedelta('-1d'),
        pd.Timedelta('59s'),
        pd.Timedelta('59m'),
        pd.Timedelta('6h'),
        pd.Timedelta('1d'),
        pd.Timedelta('7d'),
        pd.Timedelta('10y')
    ],
    labels = [
        'No difference',
        '1 minute - 1 hour',
        '1 hour - 6 hours',
        '6 hours - 1 day',
        '1 day - 1 week',
        'Greater than 1 week'
    ])
    report_time_df['time_to_report_less_than_6hrs'] = report_time_df.time_to_report <= pd.Timedelta('6h')
    report_time_df['time_to_report_greater_than_6hrs'] = report_time_df.time_to_report > pd.Timedelta('6h')
    
    return report_time_df
    

# Visualizations for Final Notebook


def viz1(top_crimes_df): 
    top_crimes_df.crime_type.value_counts().plot(kind='pie', y='cleared', autopct="%1.1f%%")
    # remove y axis label
    plt.ylabel(None)
    #add title
    plt.title('Top 10 Crimes as Percentage of Overall Crime Rate in Subset')
    plt.show()
        
def viz2(top_crimes_df, train):
    plt.title("Relationship Between Crime Type and Clearance Rate")
    top_crimes_df.groupby('crime_type').cleared.mean().sort_values(ascending=False).plot.bar()
    # calculating overall clearance rate
    clearance_rate = train.cleared.mean()
    plt.axhline(clearance_rate, label="Overall Clearance rate", linestyle = '--', c = '#45b6ef')
    plt.ylabel('Clearance Rate')
    #plt.gca().axes.get_xaxis().set_visible(False)
    plt.xticks(rotation = 90) #Rotating the xticks 35 degrees for readability
    plt.legend()
    None

def viz3(train):
    '''
    This function will take in the train dataset and return a seaborn 
    visual of the council district and their crime count in descending order
    '''
    ax = sns.countplot(data = train, y = 'council_district',order = train['council_district']
                       .value_counts(ascending = False).index, color ='lightseagreen')
    plt.xlabel('Crime Count',fontsize=14)# set up the x axis. 
    plt.ylabel('Council District',fontsize=14)# set up the y axis
    plt.title('Crime Rate by Council District',fontsize=20) # set up the title.
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    plt.show()

    return

def viz4(train):
    index = [8,6,10,5,7,2,1,3,4,9]
    df1 = pd.DataFrame(train.groupby('council_district').cleared.mean(), index = index)
    ax = df1.plot.barh()
    plt.ylabel('Council District', fontsize = 14)
    plt.xlabel('% cleared', fontsize = 14)
    plt.title('Percentage of Cleared Cases in Each District', fontsize = 20)
    plt.legend()
    ax.get_legend().remove()
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    return 

def viz5(train2):
    train2.groupby(['year', 'month']).crime_type.count().unstack(0).plot.line()
    plt.title("Crime Frequency by Year")
    plt.xlabel("Months")
    plt.ylabel("Number of Crimes")
    plt.tick_params('x', rotation=360)
    #plt.axhline(overall_mean,color="r")
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    None

def viz6(train2):
    y = train2.groupby(['weekdays','year'])['crime_type'].count()
    #Take a look at all the crime types
    train2['weekdays'] = pd.Categorical(train2['weekdays'], categories=['Monday', 'Tuesday', 'Wednesday', 
                                                                        'Thursday', 'Friday', 'Saturday','Sunday'])
    #overall_mean = df.groupby('month').crime_type.value_counts()
    #Assuming 0 = Sunday, 1 = Monday, 2 = Tuesday, 3 = Wednesday, 4 = Thursday, 5 = Friday, 6 =Saturday
    y.unstack(0).plot.bar()
    #sns.barplot(x=None, y = y, data = y, ci = None)
    plt.title("Crime Frequency by Weekday")
    plt.xlabel("Weekdays")
    plt.ylabel("Number of Crimes")
    plt.tick_params('x', rotation=360)
    #plt.axhline(overall_mean,color="r")
    None
    
def viz7(report_time_df):
    sns.barplot(data = report_time_df, x = 'time_to_report_bins', y = 'cleared', ci = None)
    plt.xticks(rotation = 15)
    plt.xlabel('Time to report', fontsize=12)
    plt.ylabel('Percentage of cases cleared', fontsize=12)
    plt.title('The sooner a crime is reported the more likely it is to be solved.', fontsize=15)
    plt.show()
    
#Statistical analysis 

def pearsonr(variable, target, alpha =.05):
    corr, p = stats.pearsonr(variable, target)
    print(f'The correlation value between the two variables is {corr:.4} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')

def t_test_ind(sample1, sample2, alternative = 'two-sided', alpha =.05):
    t, p = stats.ttest_ind(sample1, sample2, alternative = alternative)
    print(f'The t value between the two samples is {t:.4} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')
        
def t_test_1sample(sample, overall_sample, alpha =.05):
    t, p = stats.ttest_1samp(sample, overall_sample)
    print(f'The t value between the two samples is {t:.4} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p/2 < alpha:
        print('Since the P value halved is less than the alpha, we reject the null hypothesis.')
    elif t < 0:
        print("Since the T value is less than zero, we fail to reject null hypothesis")
    if p/2 > alpha:
        print("Since the P value is greater than the alpha, e fail to reject null hypothesis.")
    
    

def chi2(variable, target, alpha=.05):
    observed = pd.crosstab(variable, target)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'The chi2 value between the two variables is {chi2} and the P-Value is {p}.')
    print('----------------------------------------------------------------------------')
    if p < alpha:
        print('Since the P value is less than the alpha, we reject the null hypothesis.')
    else:
        print('Since the P value is greater than the alpha, we fail to reject the null hypothesis.')
        
def run_chi2(train, cat_var, target):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected

# catergorical vs continuous

def plot_variable_pairs(df):
    sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}})

def plot_categorical_and_continuous_vars(df, cat_var, cont_var):
    sns.barplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.boxplot(data=df, y=cont_var, x=cat_var)
    plt.show()
    sns.stripplot(data=df, y=cont_var, x=cat_var)
    plt.show()


# Univariate Exploration

def explore_univariate(train, cat_vars, quant_vars):
    for var in cat_vars:
        explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_bivariate(train, target, cat_vars, quant_vars):
    for cat in cat_vars:
        explore_bivariate_categorical(train, target, cat)
    for quant in quant_vars:
        explore_bivariate_quant(train, target, quant)

def explore_multivariate(train, target, cat_vars, quant_vars):
    '''
    '''
    plot_swarm_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    violin = plot_violin_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    pair = sns.pairplot(data=train, vars=quant_vars, hue=target)
    plt.show()
    plot_all_continuous_vars(train, target, quant_vars)
    plt.show()    


### Univariate

def plot_distributions(df):
    for col in df.columns:
        sns.histplot(x = col, data=df)
        plt.title(col)
        plt.show()

def plot_distribution(df, var):
    sns.histplot(x = var, data=df)
    plt.title(f'Distribution of {var}', fontsize=15)
    plt.show()
    
def get_box(df, cols):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = cols

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()


def explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats

def freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table


#### Bivariate

def explore_bivariate_categorical(train, target, cat_var):
    '''
    takes in categorical variable and binary target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the target. 
    '''
    print(cat_var, "\n_____________________\n")
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2_summary, observed, expected = run_chi2(train, cat_var, target)
    p = plot_cat_by_target(train, target, cat_var)

    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")

def explore_bivariate_quant(train, target, quant_var):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    mann_whitney = compare_means(train, target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = plot_boxen(train, target, quant_var)
    swarm = plot_swarm(train, target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nMann-Whitney Test:\n", mann_whitney)
    print("\n____________________\n")

## Bivariate Categorical


def plot_cat_by_target(train, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p


## Bivariate Quant

def plot_swarm(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_bar(train, cat_var, quant_var):
    average = train[quant_var].mean()
    p = sns.barplot(data=train, x=cat_var, y=quant_var, palette='Set1')
    p = plt.title(f'Relationship between {cat_var} and {quant_var}.', fontsize=15)
    p = plt.axhline(average, ls='--', color='black')
    return p

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)


### Multivariate

#Function to visualize correlations
def plot_correlations(df):
    plt.figure(figsize= (15, 8))
    df.corr()['cleared'].sort_values(ascending=False).plot(kind='bar', color = 'darkcyan')
    plt.title('Correlations with Clearance', fontsize = 18)
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.show()

def plot_all_continuous_vars(train, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing the target variable. 
    '''
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()

def plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
