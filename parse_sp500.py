import csv
import datetime

with open('sp500.data') as f, open('sp500.csv', 'w', newline='') as out:
    csv_writer = csv.writer(out)
    csv_writer.writerow(["Date", "Open", "High", "Low", "Close", "AdjClose", "Volume"])
    for idx, line in enumerate(f):
        if idx == 0:
            continue

        line = [t.strip() for t in line.split('\t')]
        line[0] = datetime.datetime.strptime(line[0], "%b %d, %Y").strftime('%Y-%m-%d')
        for i in range(1, len(line)):
            t = line[i].replace(",", "")
            if t != '-':
                line[i] = float(t)
            else:
                line[i] = -1
        csv_writer.writerow(line)

