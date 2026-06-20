import matplotlib.pyplot as plt
import numpy as np
import csv
import gzip
import json
import os
import random
import sys
import argparse

from feature_pipeline.cleaning.spcc_clean import clean_event, summarize_cleaning_results
from feature_pipeline.config import DEFAULT_GROUPING, DEFAULT_NORMALIZATION
from feature_pipeline.interpolation.spcc_legacy_reference import (
	reconstruct_augment,
	reconstruct_last_observation,
	reconstruct_spline,
)
from feature_pipeline.interpolation.spcc_native_reconstruct import reconstruct_bandwise_grid
from feature_pipeline.loaders.spcc_raw import DEFAULT_SPCC_RAW_GLOB, iter_spcc_files, load_spcc_raw_event
from feature_pipeline.policies import DEFAULT_SPCC_POLICY, KEY_TYPES

flux_norm = DEFAULT_NORMALIZATION.flux_norm
time_norm = DEFAULT_NORMALIZATION.time_norm
position_norm = DEFAULT_NORMALIZATION.position_norm
grouping = DEFAULT_GROUPING.grouping

key_types = KEY_TYPES


def event_metadata_tuple(event):
	return (
		event.survey,
		event.snid,
		event.sn_type,
		event.sim_type,
		event.sim_z,
		event.ra,
		event.decl,
		event.mwebv,
		event.hostid,
		event.hostz,
		event.spec,
	)


def parser_last_event(event):
	artifact = reconstruct_last_observation(event)
	return (*event_metadata_tuple(event), artifact.reconstructed_sequence)


def parser_spline_event(event):
	artifact = reconstruct_spline(event, grouping=grouping)
	return (*event_metadata_tuple(event), artifact.reconstructed_sequence)


def parser_augment_event(event):
	artifact = reconstruct_augment(event, grouping=grouping)
	return (*event_metadata_tuple(event), artifact.reconstructed_sequence)

def parser_last(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Replaces missing observation
	data with previous non-missing observation data - steps in data are present.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	event = load_spcc_raw_event(filename)
	return parser_last_event(event)

def parser_spline(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Flux observations are interpolated at grouped times
	and the errors are attributed to the grouped time closest to when they were actually measured.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	event = load_spcc_raw_event(filename)
	return parser_spline_event(event)

def parser_augment(filename):
	'''
	Reads and returns supernovae data into format to be read by the neural network. Flux observations and errors are grouped by time
	and any missing information is filled in with random numbers between the previous and next non-missing array elements. This can
	be run many times to augment the data and create a larger train/test set. This is the preferred method of reading data.
	* filename is a string containing the path to the supernovae light curve data
	* survey is a string containing the survey name
	* snid is an integer containing the supernova ID
	* ra is a float containing the RA of the supernova
	* dec is a float containing the Dec of the supernova
	* mwebv is a float describing the dust extinction
	* hostid is an integer containing the host galaxy ID
	* hostz is an array of floats containing the photometric redshift of the galaxy and the error on the measurement
	* spec is an array of floats containing the redshift
	* sim_type is a string containing the supernova type
	* sim_z is a float containing the redshift of the supernova
	* obs is a sequence of arrays each element containing [time since first observation,fluxes in each colourband,flux errors in each colourband]
	- Used in __main__() to read in the data
	'''
	event = load_spcc_raw_event(filename)
	return parser_augment_event(event)

if __name__ == '__main__':
	'''
	Program to preprocess supernovae data. Reads in all supernova data and writes it out to one file to
	be read in by the neural network training program.
	- Reads in files from ./data/SIMGEN_PUBLIC_DES/ which contains all light curve data.
	- Creates files in ./data/
	'''

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-p','--p', type=str, help='Parser type')
	parser.add_argument('-pr','--pr', type=str, help='File prefix')
	parser.add_argument('-na','--na', type=int, help='Number of augmentations')
	parser.add_argument('--input-glob', type=str, default=DEFAULT_SPCC_RAW_GLOB, help='Glob for raw SPCC DES_*.DAT files')
	args = parser.parse_args()

	if args.na:
		nb_augment = args.na
	else:
		nb_augment = 5

	if args.p:
		if args.p == 'augment':
			parser = parser_augment_event
		elif args.p == 'spline':
			parser = parser_spline_event 
			nb_augment = 1
		elif args.p == 'last':
			parser = parser_last_event
			nb_augment = 1
		else:
			parser = parser_augment_event
	else:
		parser = parser_augment_event

	if args.pr:
		prefix = args.pr
	else:
		prefix = ''

	input_files = iter_spcc_files(input_glob=args.input_glob)
	print('SPCC policy:', DEFAULT_SPCC_POLICY)
	print('Raw SPCC files found:', len(input_files))
	print('Raw input glob:', args.input_glob)
	print('Reconstruction source for current CSV generation: legacy_reference')
	os.makedirs('results/phase2_tier1', exist_ok=True)
	all_cleaning_results = []
	cleaned_events = []
	reconstruction_samples = []
	native_reconstruction_samples = []
	for f in input_files:
		raw_event = load_spcc_raw_event(f)
		cleaning_result = clean_event(
			raw_event,
			min_observations_per_event=DEFAULT_SPCC_POLICY.min_observations_per_event,
		)
		all_cleaning_results.append(cleaning_result)
		if cleaning_result.accepted:
			cleaned_events.append(cleaning_result.event)
			if len(reconstruction_samples) < 5:
				if args.p == 'spline':
					sample_artifact = reconstruct_spline(cleaning_result.event, grouping=grouping)
				elif args.p == 'last':
					sample_artifact = reconstruct_last_observation(cleaning_result.event)
				else:
					sample_artifact = reconstruct_augment(cleaning_result.event, grouping=grouping)
				reconstruction_samples.append({
					"snid": cleaning_result.event.snid,
					"implementation": "legacy_reference",
					"mode": sample_artifact.mode,
					"raw_sequence": sample_artifact.raw_sequence,
					"reconstructed_sequence": sample_artifact.reconstructed_sequence,
				})
				native_artifact = reconstruct_bandwise_grid(cleaning_result.event)
				native_reconstruction_samples.append({
					"snid": cleaning_result.event.snid,
					"implementation": "native_owned",
					"mode": native_artifact.mode,
					"notes": native_artifact.notes,
					"raw_sequence": native_artifact.raw_sequence,
					"reconstructed_sequence": native_artifact.reconstructed_sequence,
				})

	for i in range(1,nb_augment+1):

		print('Processing augmentation: ',i)

		if prefix:
			fhost = open('data/'+prefix+'_unblind_hostz_'+str(i)+'.csv', 'w')
			fnohost = open('data/'+prefix+'_unblind_nohostz_'+str(i)+'.csv', 'w')
		else:
			fhost = open('data/unblind_hostz_'+str(i)+'.csv', 'w')
			fnohost = open('data/unblind_nohostz_'+str(i)+'.csv', 'w')
		whost = csv.writer(fhost)
		wnohost = csv.writer(fnohost)
		
		sn_types = {}
		nb_sn = 0

		for cleaned_event in cleaned_events:	
			survey, snid, sn_type, sim_type, sim_z, ra, decl, mwebv, hostid, hostz, spec, obs = parser(cleaned_event)
			try:
				unblind = [sim_z, key_types[sim_type]]
			except:
				print('No information for', snid)
				continue
			for o in obs:
				whost.writerow([snid,o[0],ra,decl,mwebv,hostz[0]] + o[1:9] + unblind)
				wnohost.writerow([snid,o[0],ra,decl,mwebv] + o[1:9] + unblind)
			try:
				sn_types[unblind[1]] += 1
			except:
				sn_types[unblind[1]] = 0
			nb_sn += 1

		fhost.close()
		fnohost.close()

	cleaning_summary = summarize_cleaning_results(all_cleaning_results)
	cleaning_summary['policy'] = DEFAULT_SPCC_POLICY.__dict__
	cleaning_summary['raw_file_count'] = len(input_files)
	cleaning_summary['raw_input_glob'] = args.input_glob
	cleaning_summary['current_csv_generation_reconstruction'] = 'legacy_reference'
	cleaning_summary['owned_native_reconstruction_available'] = True
	with open('results/phase2_tier1/spcc_cleaning_report.json', 'w') as handle:
		json.dump(cleaning_summary, handle, indent=2)
	with open('results/phase2_tier1/spcc_reconstruction_samples.json', 'w') as handle:
		json.dump(reconstruction_samples, handle, indent=2)
	with open('results/phase2_tier1/spcc_native_reconstruction_samples.json', 'w') as handle:
		json.dump(native_reconstruction_samples, handle, indent=2)
		
	print('Num train: ', nb_sn)
	print('SN types: ', sn_types)
