import re
def match_lines_with_IID(file_path, prefix):
    matching_old2new_IID = {}

    with open(file_path, 'r') as file:
        item_pos = 1
        for line in file.readlines():
            # Strip any leading or trailing whitespace from the line
            line = line.strip().split('\t')
            if line[0] == 'item_id:token':
                continue
            else:
                matching_old2new_IID[line[0]] = prefix[0] + str(item_pos) + prefix[1]
                item_pos += 1
    return matching_old2new_IID

def replace_file_itemID(file_path, file_path_new, oldIID_newIID_dict, mode=0):
    fout = open(file_path_new, "w")
    with open(file_path, 'r') as file:
        for line in file.readlines():
            # Strip any leading or trailing whitespace from the line
            line = line.strip().split('\t')
            if line[mode] != 'item_id:token':
                line[mode] = oldIID_newIID_dict[line[mode]]
            fout.write('\t'.join(line)+'\n')


# Example usage
file_name = 'ml-1m'
file_item_old = './dataset/' + file_name + '/' + file_name + '.item'
file_inter_old = './dataset/' + file_name + '/' + file_name + '.inter'
file_item_new = './dataset/' + file_name + '/' + file_name + '.item_new'
file_inter_new = './dataset/' + file_name + '/' + file_name + '.inter_new'
prefixes = ['<', '>']

oldIID_newIID_dict = match_lines_with_IID(file_item_old, prefixes)

# Print the matching lines
replace_file_itemID(file_item_old, file_item_new, oldIID_newIID_dict, mode=0)
replace_file_itemID(file_inter_old, file_inter_new, oldIID_newIID_dict, mode=1)



    