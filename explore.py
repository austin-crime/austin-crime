import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

# Visualizations for Final Notebook

# Create a list with the top 10 crimes
top_10_crimes = ['DWI', 'ASSAULT WITH INJURY', 'THEFT BY SHOPLIFTING', 'HARASSMENT', 
                 'AUTO THEFT', 'ASSAULT W/INJURY-FAM/DATE VIOL', 'CRIMINAL MISCHIEF', 
                 'FAMILY DISTURBANCE', 'THEFT', 'BURGLARY OF VEHICLE']

fontsize = 20

# Create a subsetted df that only uses the top 10 crimes
def subset_top_crimes(train):
    '''
    This function will take in a dataframe, use the top 10 crimes list above and 
    return the new dataframe with all the records with just the list of crimes above.
    '''
    top_crimes_df = train.copy()
    top_crimes_df['crime_type'] = np.where(top_crimes_df.crime_type.isin(top_10_crimes), top_crimes_df.crime_type, 'OTHER')
    return top_crimes_df

def plot_cleared(df):
    df.cleared.value_counts(normalize = True).plot.barh()

    plt.title("Distribution of Clearance")

    plt.ylabel('')
    plt.xlabel('')

    plt.yticks([False, True], labels = ['Not Cleared', 'Cleared'])

    plt.gca().xaxis.set_major_formatter('{:.0%}'.format)

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

def create_time_series_df(train):
    '''
    This function will create a copy of a train dataframe and then
    set the occurence date as the index in order to create new feature for weekdays, months and years
    and then change weekdays and months into categorical variables
    '''
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
    '''
    This function will create 2 subset of data: 1 will be just friday only 
    and the other is exclude friday 
    '''
    friday_only = ['Friday']
    subset_friday = time_series_df.copy()
    subset_friday = subset_friday[subset_friday.weekdays.isin(friday_only)]
    subset_not_friday = time_series_df.copy()
    subset_not_friday = subset_not_friday[~subset_not_friday.weekdays.isin(friday_only)]#Without friday
    
    return subset_friday, subset_not_friday

# Create a dataframe that is prepare for time-series analysis on crime reporting time

def create_report_time_df(train):
    '''
    This function will take in a dataframe and then create 6 bins base on time_to_report
    '''
    # Calculate a time_to_report feature
    report_time_df = train.copy()
    report_time_df['time_to_report'] = report_time_df.report_time - report_time_df.occurrence_time

    report_time_df['time_to_report_bins'] = pd.cut(
        report_time_df.time_to_report,
        [
            pd.Timedelta('-1d'),
            pd.Timedelta('5h'),
            pd.Timedelta('10y')
        ],
        labels = [
            'Less than 5 hours',
            'Greater than 5 hours'
        ]
    )
    
    return report_time_df
    

# Visualizations for Final Notebook

color_map = [
    'lime',
    'lightcoral',
    'springgreen',
    'mistyrose',
    'darkgreen',
    'maroon',
    'olive',
    'tomato',
    'forestgreen',
    'indianred',
    'grey'
]

def viz1(top_crimes_df): 
    '''
    This function will create a percentage pie chart to show top 10 crimes
    '''
    top_crimes_df.crime_type.value_counts().plot(kind = 'pie', y = 'cleared', autopct = "%1.0f%%", colors = color_map)
    # remove y axis label
    plt.ylabel(None)
    #add title
    plt.title('Top 10 Crimes as Percentage of Overall Crime Rate in Subset', fontsize = fontsize)
    plt.show()
        
    
def viz2(top_crimes_df, train):
    '''
    This fucntion will create a bar chart to visual the relationship between target and the features
    '''
    # Calculate overall clearance rate
    clearance_rate = train.cleared.mean()
    plt.title("Relationship Between Crime Type and Clearance Rate", fontsize = fontsize)
    sns.barplot(x="cleared", y="crime_type", data=top_crimes_df,
            order=['DWI', 'ASSAULT W/INJURY-FAM/DATE VIOL', 'THEFT BY SHOPLIFTING', 'AUTO THEFT', 
                   'ASSAULT WITH INJURY', 'THEFT', 'CRIMINAL MISCHIEF', 'FAMILY DISTURBANCE', 
                   'BURGLARY OF VEHICLE', 'HARASSMENT', 'OTHER'], color ='royalblue', ci=None)
    plt.axvline(clearance_rate, label="Overall Clearance rate", linestyle = '--', alpha=.8, color='orange')
    plt.ylabel('')
    plt.xlabel('Clearance Rate')
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    plt.legend()
    plt.show()

def viz3(train):
    '''
    This function will take in the train dataset and return a seaborn 
    visual of the council district and their crime count in descending order
    '''
    fig, ax = plt.subplots(ncols = 2, nrows = 1)

    # sns.countplot(data = train, y = 'council_district',order = train['council_district']
    #                    .value_counts(ascending = False).index, color ='royalblue', ax = ax[0])
    # plt.xlabel('Crime Count')# set up the x axis. 
    # plt.ylabel('Council District')# set up the y axis
    # plt.title('Crime Rate by Council District', fontsize = fontsize) # set up the title.
    # plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)

    # index = [8,6,10,5,7,2,1,3,4,9]
    # df1 = pd.DataFrame(train.groupby('council_district').cleared.mean())
    # df1.plot.barh(ax = ax[1])
    # plt.ylabel('Council District')
    # plt.xlabel('% cleared')
    # plt.title('Percentage of Cleared Cases in Each District', fontsize = fontsize)
    # plt.legend()
    # ax[1].get_legend().remove()
    # plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    # plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))

    train.groupby('council_district').cleared.count().sort_values().plot.barh(ax = ax[0])
    ax[0].set(xlabel = 'Crime Count')
    ax[0].set(ylabel = 'Council District')
    ax[0].set_title('Crime Total by Council District', fontsize = fontsize)
    ax[0].grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)

    train.groupby('council_district').cleared.mean().reindex(index = [8, 10, 6, 5, 2, 1, 7, 4, 3, 9]).plot.barh(ax = ax[1])
    ax[1].set(xlabel = '% Cleared')
    ax[1].set(ylabel = 'Council District')
    ax[1].set_title('Clearance Rate by Council District', fontsize = fontsize)
    ax[1].grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    ax[1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))

    plt.show()

    return

def viz4(train):
    '''
    This function will create a horizontal bar chart to visual 
    the percentage of crimes in each council districts
    '''
    index = [8,6,10,5,7,2,1,3,4,9]
    df1 = pd.DataFrame(train.groupby('council_district').cleared.mean(), index = index)
    ax = df1.plot.barh()
    plt.ylabel('Council District')
    plt.xlabel('% cleared')
    plt.title('Percentage of Cleared Cases in Each District', fontsize = fontsize)
    plt.legend()
    ax.get_legend().remove()
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter('{:.0%}'.format))
    return 

def viz5(train2):
    '''
    This function will create a line plot base on year and month 
    and total number of crime types
    '''
    train2.groupby(['year', 'month']).crime_type.count().unstack(0).plot.line()
    plt.title("Crime Frequency by Year", fontsize = fontsize)
    plt.xlabel("")
    plt.ylabel("Number of Crimes")
    plt.tick_params('x', rotation=360)
    plt.legend(bbox_to_anchor= (1.16,1))
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.5, alpha= 0.8)
    None

def viz6(train2):
    '''
    This function will create a bar chart to visualize the total number of crimes 
    and then put the weekdays into categorical so the chart in order from Monday to Sunday
    '''
    y = train2.groupby(['weekdays','year'])['crime_type'].count()
    #Take a look at all the crime types
    y.unstack(0).plot.bar()
    #sns.barplot(x=None, y = y, data = y, ci = None)
    plt.title("Crime Frequency by Weekdays", fontsize = fontsize)
    plt.xlabel("Years")
    plt.ylabel("Number of Crimes")
    plt.tick_params('x', rotation=360)
    plt.legend(bbox_to_anchor= (1.03,1))
    None
    
def viz7(report_time_df):
    '''
    This fucntion will create a barplot for each bin of time to report
    '''
    sns.barplot(data = report_time_df, x = 'time_to_report_bins', y = 'cleared', ci = None)
    
    plt.xlabel('Time to report')
    plt.ylabel('Percentage of cases cleared')
    plt.title('The sooner a crime is reported the more likely it is to be solved.', fontsize = fontsize)
    plt.gca().yaxis.set_major_formatter('{:.0%}'.format)
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
    This function will create 3 plot: swram plot, violin plot and pairplot
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
    '''
    This function will create a histogram for each item in a column
    '''
    for col in df.columns:
        sns.histplot(x = col, data=df)
        plt.title(col)
        plt.show()

def plot_distribution(df, var):
    '''
    This function will create a histogram too look overall 
    '''
    sns.histplot(x = var, data=df)
    plt.title(f'Distribution of {var}')
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
    '''
    Create a barplot
    '''
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p


## Bivariate Quant

def plot_swarm(train, target, quant_var):
    '''
    Create a swram plot
    '''
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_boxen(train, target, quant_var):
    '''
    Create a boxen plot
    '''
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def plot_bar(train, cat_var, quant_var):
    '''
    Create a barplot
    '''
    average = train[quant_var].mean()
    p = sns.barplot(data=train, x=cat_var, y=quant_var, palette='Set1')
    p = plt.title(f'Relationship between {cat_var} and {quant_var}.')
    p = plt.axhline(average, ls='--', color='black')
    return p

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def compare_means(train, target, quant_var, alt_hyp='two-sided'):
    '''
    This function will peforma manwhitenyu stat test 
    '''
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)


### Multivariate

#Function to visualize correlations
def plot_correlations(df):
    '''
    This function will compare correlations between the target and features
    '''
    plt.figure(figsize= (15, 8))
    df.corr()['cleared'].sort_values(ascending=False).plot(kind='bar', color = 'darkcyan')
    plt.title('Correlations with Clearance')
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
    '''
    This fucntion will create a violin plot
    '''
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        # Enumerate() method adds a counter to an iterable 
        # and returns it in a form of enumerating object.
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    '''
    This function will create a swram plot
    '''
    cols = len(cat_vars) #Checking the length 
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        # Enumerate() method adds a counter to an iterable 
        # and returns it in a form of enumerating object.
        for i, cat in enumerate(cat_vars): 
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
