# Copyright © 2021-2025 Eduardo Vieira de Souza
# Copyright © 2021-2025 Adriana Canedo
# Copyright © 2021-2025 Cristiano Valim Bizarro
#
# This file is part of uProteInS.
#
# uProteInS is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# uProteInS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# uProteInS. If not, see <https://www.gnu.org/licenses/>.


import pandas as pd
import os


class SpecMiner(object):
    def __init__(self, results, pin_dir):
        self.df = pd.read_csv(results, sep='\t')
        self.pinFolder = pin_dir
        self.df = self.df.drop_duplicates()
        # self.fixedFileNames = self.__fix_filenames()
        self.fixedFileNames = self.df["SpecId"]
        self.scanNums = self.df["ScanNum"].tolist()
        self.extendedORFs = self.df["Extended ORF"].tolist()
        self.extendedSequences = self.df["Extended Sequence"].tolist()

    def __fix_filenames(self):
        """ legacy code """
        names = self.df["SpecFile"].tolist()
        fixed = [name[:-5] for name in names]
        self.df.insert(0, "FixedFiles", fixed)
        return fixed

    def mine_spectra(self, output):
        files = os.listdir(self.pinFolder)
        i = 0
        extended_orfs = []
        extended_seqs = []
        frames = []
        cols = None
        for scan, file in zip(self.scanNums, self.fixedFileNames):
            for file in files:
                full_df = pd.read_csv(f'{self.pinFolder}/{file}', sep='\t')
                if cols is None:
                    cols = full_df.columns
                intersection = full_df[(full_df["ScanNr"] == scan) & (full_df["SpecId"].str.contains(file, na=False))]
                frames.append(intersection)
                scans = intersection["ScanNr"].tolist()
                for _ in range(len(scans)):
                    extended_orfs.append(self.extendedORFs[i])
                    extended_seqs.append(self.extendedSequences[i])
            print(i)
            i += 1

            if i == 5:
                break

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols if cols is not None else [])
        df.insert(3, "Extended ORF", extended_orfs)
        df.insert(4, "Extended Sequence", extended_seqs)
        df.to_csv(output, sep='\t', index=False)


class MinePredicted(object):
    def __init__(self, predicted, results):
        self.predicted = pd.read_csv(predicted, sep='\t')
        self.df = pd.read_csv(results, sep='\t')

        self.specFiles = self.df["SpecFile"].tolist()
        self.scans = self.df["ScanNum"].tolist()

        self.highConfidenceScans = self.predicted["ScanNr"].tolist()
        self.highConfidenceFiles = self.predicted["SpecId"].tolist()

    def filter(self, output):
        high_scans = []
        high_files = []
        for file, scan in zip(self.specFiles, self.scans):
            for i in range(len(self.highConfidenceFiles)):
                if file[:-5] in self.highConfidenceFiles[i] and self.highConfidenceScans[i] == scan:
                    high_scans.append(scan)
                    high_files.append(file)
        frames = [
            self.df[(self.df["SpecFile"] == file) & (self.df["ScanNum"] == scan)]
            for file, scan in zip(high_files, high_scans)
        ]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=self.df.columns)
        df = df.drop_duplicates()
        df.to_csv(output, sep='\t', index=False)

    def filter_v2(self):
        hcs = self.__get_hc()



    def __get_hc(self):
        hcs = {}  # spectra predicted to have high confidence by the ML model
        for file, scan in zip(self.highConfidenceFiles, self.highConfidenceScans):
            if file in hcs:
                hcs[file] += f',{scan}'
            else:
                hcs[file] = f'{scan}'
        return hcs


class SpectrumMiner(object):
    def __init__(self, results, predicted, testing=False):
        self.results = pd.read_csv(results, sep='\t')
        self.testing = testing
        self.__check_testing()
        self.unfilteredScans = self.results["ScanNum"].tolist()
        self.__rename()

        self.predicted = pd.read_csv(predicted, sep='\t')


        self.highConfidenceScans = self.predicted["ScanNr"].tolist()
        self.highConfidenceFiles = self.predicted["SpecId"].tolist()
        self.__rename_predicted()
        self.fixedPredictedFiles = self.predicted["RenamedFiles"].tolist()
        self.HCs = self.__get_hc()

    def __check_testing(self):
        if self.testing:
            ...

    def __get_hc(self):
        hcs = {}  # spectra predicted to have high confidence by the ML model
        for file, scan in zip(self.fixedPredictedFiles, self.highConfidenceScans):
            if file in hcs:
                hcs[file].append(scan)
            else:
                hcs[file] = [scan]
        return hcs

    def __rename_predicted(self):
        renamed = ['_'.join(file.split("_")[:-6]) for file in self.highConfidenceFiles]
        self.predicted.insert(0, 'RenamedFiles', renamed)

    def __rename(self):
        spec_files = self.results["SpecFile"].tolist()
        renamed = []
        for file, scan in zip(spec_files, self.unfilteredScans):
            renamed.append(file[:-5])
        self.results.insert(0, "RenamedFiles", renamed)

    def filter_results(self, output):
        frames = [
            self.results[(self.results["ScanNum"].astype('int64').isin(self.HCs[hc])) & (self.results["RenamedFiles"] == hc)]
            for hc in self.HCs
        ]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=self.results.columns)
        df = df.drop_duplicates()
        df.to_csv(f'{output}', sep='\t', index=False)

    def add_lc_spectra_for_hc_orf(self, filetype, folder):
        """ This adds all spectra for each ORF that has at least one HC spectrum """
        self.__check_results_dir(folder)
        four = pd.read_csv(f'{folder}/post_perc/{filetype}_results_04.txt', sep='\t')
        five = pd.read_csv(f'{folder}/post_perc/{filetype}_results_05.txt', sep='\t')
        hc_orfs = five["Extended Sequence"].tolist()
        four_passed = four[four["Extended Sequence"].isin(hc_orfs)]
        four_passed.to_csv(f'{folder}/Results/{filetype}_results.txt', sep='\t', index=False)


    @staticmethod
    def __check_results_dir(folder):
        if not os.path.exists(f"{folder}/Results"):
            os.system(f'mkdir {folder}/Results')


# data = SpectrumMiner(predicted='../for_predicting/predicted_pin_files.txt', results='Transcriptome/transcriptome_results_04.txt')
# data.filter_results('Transcriptome/post_perc/transcriptome_results_05.txt')

# data = SpectrumMiner(predicted='../for_predicting/predicted_pin_files_supervised.txt', results='../post_perc/genome_results_05.txt')
# rna_data = SpectrumMiner(predicted='../for_predicting/Transcriptome/predicted_pin_files_semisupervised.txt', results='Transcriptome/transcriptome_results_04.txt')
# data ='../for_predicting/genome_predicted.txt', results='Genome/genome_results_04.txt')
# data = SpectrumMiner(results='df2', predicted='df1')
# data.filter_results()
# rna_data.filter_results('Transcriptome/transcriptome_orfs_semisupervised_24_06')

# mtb_exosomes = SpectrumMiner(predicted='predicted/exosomes_mtb_predicted.txt', results='u')