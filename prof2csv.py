"""
File: prof2csv.py

This programs reads prof_dgs.csv to a datafram and writes a new file, dgs_pred_ui.prof, with the following changes:
The format of the dgs_prod_ui.prof file is:
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    1                                           def prof2csv():
    2         1         10.0     10.0      0.0      import pandas as pd
    3         1         10.0     10.0      0.0      df = pd.read_csv('prof_dgs.csv')
    4         1         10.0     10.0      0.0      df.to_csv('prof_dgs2.csv')

Split each line into separate columns.
"""
import pandas as pd
import re

# Create a compiled regular expression pattern that separates the fields Line #, Hits, Time, Per Hit, % Time
pattern = re.compile(r'(\s*\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)(.*)')

with open('dgs_pred_ui.prof', 'r') as in_file:
    lines = in_file.readlines()

    line_dict = {}

    for line in lines:
        match line:
            case '\n':
                continue
            case 'Line #      Hits         Time  Per Hit   % Time  Line Contents\n':
                continue
            case '==============================================================\n':
                continue

        matches = pattern.match(line)

        if matches is not None:
            # Place all matches into a list
            flds = matches.groups()
            # print(f'Line #: {flds[0]}, Hits: {flds[1]}, Time: {flds[2]}, Per Hit: {flds[3]}, % Time: {flds[4]}, Line Contents: {flds[5]}')
            line_dict[flds[0]] = flds[1:]

    # Make like_dict into a list of lists where key is the first element of the list
    line_list = [[k] + list(v) for k, v in line_dict.items()]
    df = pd.DataFrame(line_list, columns=['Line #', 'Hits', 'Time', 'Per Hit', '% Time', 'Line Contents'])
    # Convert the Line # to an integer
    df['Line #'] = df['Line #'].astype(int)
    # Covert Hist to an integer
    df['Hits'] = df['Hits'].astype(int)
    # Convert Time to a float
    df['Time'] = df['Time'].astype(float)
    # Convert Per Hit to a float
    df['Per Hit'] = df['Per Hit'].astype(float)
    # Convert % Time to a float
    df['% Time'] = df['% Time'].astype(float)
    df.to_csv('dgs_pred_ui.csv', index=False)