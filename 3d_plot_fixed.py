bins = [20, 25, 30, 35, 40, 45, 50]
labels = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Now, let's group the df by the three categorical columns and calculate the average decision (probability of hiring)
grouped_data_uni_degree_age = df.groupby(['ind-university_grade', 'ind-degree', 'age_group'])['decision'].mean().reset_index()
grouped_data_uni_degree_age.rename(columns={'decision': 'probability_of_hiring'}, inplace=True)

# Get the unique values of the categories
uni_grades = df['ind-university_grade'].unique()
degrees = df['ind-degree'].unique()
age_groups = labels

# Now, we can plot the 3D bar chart
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Set up color map
color_map = plt.get_cmap('viridis', len(uni_grades))

# Loop through the df to plot the bars
for i, uni_grade in enumerate(uni_grades):
    for j, degree in enumerate(degrees):
        for k, age_group in enumerate(age_groups):
            subset = grouped_data_uni_degree_age[
                (grouped_data_uni_degree_age['ind-university_grade'] == uni_grade) &
                (grouped_data_uni_degree_age['ind-degree'] == degree) &
                (grouped_data_uni_degree_age['age_group'] == age_group)
            ]
            if not subset.empty:
                x = i
                y = j
                z = k
                dx = 0.4
                dy = 0.4
                dz = subset['probability_of_hiring'].values[0]
                color = color_map(i)
                ax.bar3d(x, y, z, dx, dy, dz, color=color, alpha=0.7, shade=True)

# Labeling and aesthetics
ax.set_xlabel('University Grade')
ax.set_ylabel('Degree')
ax.set_zlabel('Age Group')
ax.set_xticks(range(len(uni_grades)))
ax.set_yticks(range(len(degrees)))
ax.set_zticks(range(len(age_groups)))
ax.set_xticklabels(uni_grades, rotation=45)
ax.set_yticklabels(degrees, rotation=45)
ax.set_zticklabels(age_groups)
ax.set_title('3D Bar Chart of Probability of Being Hired Based on University Grade, Degree, and Age Group')

# Show the plot
plt.show()