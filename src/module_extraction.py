import json, re, tempfile, subprocess, os
import numpy as np

def module_extraction(example_code, format_code=False):
	if format_code:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".v") as fp:
			fp.write(example_code.encode(encoding='utf-8'))
			fp.close()
			sub_run = subprocess.run(['verible-verilog-format', fp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			
			# example_code = sub_run.stdout.decode("utf-8")
			with open(fp.name, 'r') as file:
				example_code = file.read()
	
		subprocess.run(['rm', '-f', fp.name])
	code_lines = example_code.splitlines()
	regex_stage = 0
	regexes = [r'^.*module\s+', r'^.*module$']
	
	module_def_span = np.array([None, None, None, None])
	num_indexs = 0
	comment_stage = 0
	multiline_comment = [False, -1]
	all_comment_ranges = []
	for i in range(len(code_lines)):
		line = code_lines[i]
		num_indexs += (i > 0) # plus previous \n
		if comment_stage == 0:
			line_for_comment = str(line)
			comment_ranges = []
			comment_num_indexs = 0
			while True:
				double_slash_index = line_for_comment.find("//")
				double_slash_index = [double_slash_index, double_slash_index]
				
				single_slash_index = line_for_comment.find("/*")
				single_slash_index = [single_slash_index, single_slash_index]
				
				single_slash_end_index = line_for_comment.rfind("*/")
				single_slash_end_index = [single_slash_end_index, single_slash_end_index]

				#
				# Handle for case of multi-line comment
				if multiline_comment[0]:
					if (double_slash_index[0] != -1 and single_slash_end_index[0] != -1):
						if (double_slash_index[0] < single_slash_end_index[0]) or (double_slash_index[0] - single_slash_end_index[0] == 1):
							double_slash_index[0] = -1
						# else: No else!!!							
				else:
					if double_slash_index[0] != -1 and single_slash_index[0] != -1: # fix all cases of comment!!
						if double_slash_index[0] < single_slash_index[0]:
							single_slash_index[0] = -1
						else:
							double_slash_index[0] = -1
					if (double_slash_index[0] != -1 and single_slash_end_index[0] != -1):
						if double_slash_index[0] < single_slash_end_index[0]:
							single_slash_end_index[0] = -1
					if single_slash_end_index[0] > -1 and single_slash_index[0] > -1 and (single_slash_end_index[0] - single_slash_index[0] == 1):
						single_slash_end_index[0] = single_slash_end_index[1] = -1
				#
				# handle for start comment
				# start_comment_index = double_slash_index if double_slash_index != None else single_slash_index if single_slash_index != None else None
				if multiline_comment[0]:
					if single_slash_end_index[0] != -1:
						multiline_comment[0] = False

						len_of_old_line = len(line_for_comment)
						comment_ranges.append([0, single_slash_end_index[0] + 1])
						comment_ranges[-1][0] += comment_num_indexs
						comment_ranges[-1][1] += comment_num_indexs
						
						line_for_comment = line_for_comment[single_slash_end_index[0]+2:]
						comment_num_indexs += len_of_old_line - len(line_for_comment)
					elif multiline_comment[1] != i:
						len_of_old_line = len(line_for_comment)
						if len_of_old_line:
							comment_ranges.append([0, len_of_old_line - 1])
							comment_ranges[-1][0] += comment_num_indexs
							comment_ranges[-1][1] += comment_num_indexs
							
							line_for_comment = line_for_comment[:0]
							comment_num_indexs += len_of_old_line - len(line_for_comment)
						break
				else:
					if single_slash_end_index[0] != -1:
						if (double_slash_index[0] != -1 and double_slash_index[0] > single_slash_end_index[0]) or (single_slash_index[0] != -1 and single_slash_index[0] > single_slash_end_index[0]) or (double_slash_index[0] == -1 and single_slash_index[0] == -1):
							raise Exception(f"Error end of multiline comment!!!\n{example_code}")
				
				if not multiline_comment[0]:
					if double_slash_index[0] != -1:
						len_of_old_line = len(line_for_comment)
						comment_ranges.append([double_slash_index[0], len_of_old_line - 1])
						comment_ranges[-1][0] += comment_num_indexs
						comment_ranges[-1][1] += comment_num_indexs
						
						line_for_comment = line_for_comment[:double_slash_index[0]]
						comment_num_indexs += len_of_old_line - len(line_for_comment)
					if single_slash_index[0] != -1 and single_slash_end_index[0] == -1:
						multiline_comment[0] = True
						multiline_comment[1] = i

						len_of_old_line = len(line_for_comment)
						comment_ranges.append([single_slash_index[0], len_of_old_line - 1])
						comment_ranges[-1][0] += comment_num_indexs
						comment_ranges[-1][1] += comment_num_indexs
						
						line_for_comment = line_for_comment[:single_slash_index[0]]
						comment_num_indexs += len_of_old_line - len(line_for_comment)
						
					elif single_slash_index[0] != -1 and single_slash_end_index[0] != -1:
						len_of_old_line = len(line_for_comment)
						comment_ranges.append([single_slash_index[0], single_slash_end_index[0]+1])
						comment_ranges[-1][0] += comment_num_indexs
						comment_ranges[-1][1] += comment_num_indexs
						
						line_for_comment = line_for_comment[:single_slash_index[0]] + line_for_comment[single_slash_end_index[0]+2:]
						comment_num_indexs += len_of_old_line - len(line_for_comment)
						
				if double_slash_index[1] != -1 or single_slash_index[1] != -1 or single_slash_end_index[1] != -1:
					continue
				else:
					break
				
		# if len(comment_ranges):
		#	 print(comment_ranges, i)

		section_start_idx = 0
		cur_section = ''
		all_comment_ranges.append(comment_ranges[:])
		if len(comment_ranges): # sort for correct order
			comment_ranges.sort(key=lambda item: item[0])
			
		while section_start_idx < len(line):
			old_section_start_idx = section_start_idx
			if len(comment_ranges):
				cur_section = line[section_start_idx:comment_ranges[0][0]]
				section_start_idx += comment_ranges[0][1] + 1
				comment_ranges.pop(0)
			else:
				cur_section = line[section_start_idx:]
				section_start_idx += len(cur_section)
		
			#
			if regex_stage == 0:
				for regex in regexes:
					module_match = re.match(regex, cur_section)
					if module_match != None:
						regex_stage += 1
						span = module_match.span()
						module_def_span = np.vstack((module_def_span, [span[0] + num_indexs, span[1] + num_indexs, None, None]))
						break
			if regex_stage == 1:
				semicon_pos = cur_section.find(";")
				if semicon_pos != -1:
					module_def_span[-1][-2] = num_indexs + semicon_pos + 1
					regex_stage += 1
			if regex_stage == 2:
				find_start = 0
				while True:
					endmodule_word = "endmodule"
					endmodule_pos = cur_section.find(endmodule_word, find_start)
					endmodule_pos_end = endmodule_pos + len(endmodule_word)
					if endmodule_pos != -1:
						left_endmodule = False
						right_endmodule = False

						#
						if endmodule_pos == 0:
							left_endmodule = True
						elif endmodule_pos > 0:
							left_ord = ord(cur_section[endmodule_pos - 1] )
							if not ((left_ord >= 65 and left_ord <= 90) or (left_ord >= 97 and left_ord <= 122) or left_ord == 95):
								left_endmodule = True

						#
						if endmodule_pos_end < len(cur_section):
							right_ord = ord(cur_section[endmodule_pos_end])
							if not ((right_ord >= 65 and right_ord <= 90) or (right_ord >= 97 and right_ord <= 122) or right_ord == 95):
								right_endmodule = True
						elif endmodule_pos_end == len(cur_section):
							right_endmodule = True

						if left_endmodule and right_endmodule:
							module_def_span[-1][-1] = num_indexs + endmodule_pos + 1
							regex_stage = 0
							break
						else:
							find_start += endmodule_pos_end
					else:
						break
							# if left_ord == 13 or left_ord == 10 or left_ord == 9 or left_ord == 12 or left_ord == 32:
							#	 left_endmodule = True
						# module_def_span[-1][-1] = num_indexs + endmodule_pos + 1
						# regex_stage = 0
	
			num_indexs += section_start_idx - old_section_start_idx
	if len(module_def_span.shape) < 2:
		error_txt = f"Module span is all null {len(example_code)}: {example_code}"
		
		raise Exception(error_txt)
		
	module_def_span = np.delete(module_def_span, 0, 0)
	module_wrapper = ""
	module_definition = []
	module_output_code = []
	for span in module_def_span:
		if None in span:
			raise Exception(f"Span Error!! {len(example_code)}: {example_code}")
		module_definition += [example_code[span[0]:span[2]]]
		module_output_code += [example_code[span[2]:span[3]+8]]
		
	module_definition_with_upper_comments = []
	for i in range(len(module_def_span)):
		span = module_def_span[i]
		if i == 0:
			span[0] = 0
		else:
			span[0] = module_def_span[i-1][-1] + 1
		module_definition_with_upper_comments += [example_code[span[0]:span[2]]]
		
	return (module_definition, module_output_code, module_definition_with_upper_comments, module_def_span, all_comment_ranges)

logic_keywords = ["always", "always_comb",	"and", "assign",
				  "not", "nand", "nor",  "or", 
				  # "pull0","pull1",	"strong0","strong1","supply0","supply1",
				  # "weak0", "weak1",
				  "xnor","xor", "display"]

def is_module_contain_logic(cur_code, all_comment_ranges=None):
	# cur_code = dataset_no_logic['code'][i]
	# cur_code = cur_code.replace("\\\\", "\\")

	if all_comment_ranges == None:
		all_comment_ranges = module_extraction(cur_code)
		all_comment_ranges = all_comment_ranges[-1]

	cur_code_lines = cur_code.splitlines()
	final_code = ""
	for ii in range(len(cur_code_lines)):
		cur_code_line = np.array(list(cur_code_lines[ii]))
		try:
			cur_comment_ranges = all_comment_ranges[ii]
		except:
			# raise Exception(f'Error with code: {cur_code}, \n Comment: {all_comment_ranges}')
			return (None, None)

		filter_range = [i for i in range(len(cur_code_line))]

		for comment_range in cur_comment_ranges:
			if len(comment_range):
				start_range, end_range = comment_range
				if end_range == None:
					end_range = len(cur_code_line)
			else:
				continue

			for iii in range(start_range, end_range):
				if iii in filter_range:
					filter_range.remove(iii)
		final_code += ''.join(cur_code_line[filter_range]) + "\n"
	
	found_logic_keyword = False

	for keyword in logic_keywords:
		match = re.search(r"\b" + re.escape(keyword) + r"\b", final_code)
		if match:
			found_logic_keyword = True
			break
	return found_logic_keyword, final_code
	# if not found_logic_keyword:
	#	 # no_logic_index.append(i)
	#	 # return dataset_no_logic["big_idx"][i]
	#	 return i
	# return None