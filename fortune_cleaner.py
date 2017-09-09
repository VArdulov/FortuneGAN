from csv import DictWriter

fortune_text = []
if __name__ == '__main__':
    with open('fortunes.txt', 'r') as f_file:
        fortune_text = f_file.readlines()

    csv_entries = []
    author = 'Uknown'
    running_text = ''
    for line in fortune_text:
        if line[0] != '%':
            if '~' not in line and '\t' not in line and '        ' not in line:
                # actual quote text
                running_text += (" " + line.rstrip())
        else:
            csv_entries.append({'fortune_text': running_text})
            running_text = ''

    with open('data/fortune_text.csv', 'w') as csv_file:
        writer = DictWriter(csv_file, fieldnames=['fortune_text'])
        writer.writerows(csv_entries)
    print('All Done!')



