
import re
from collections import defaultdict
from re import RegexFlag
from typing import List



def extract_structure_data(plain_text_content: str):
    # extracts lines starts with specific flags
    # map id to its related information
    data = []

    predict_outputs = re.findall("^D.+", plain_text_content, RegexFlag.MULTILINE)
    ground_outputs = re.findall("^T.+", plain_text_content, RegexFlag.MULTILINE)
    source_inputs = re.findall("^S.+", plain_text_content, RegexFlag.MULTILINE)

    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        try:
            predict_id, _, predict_clean = predict.split('\t')
            ground_id, ground_clean = ground.split('\t')
            source_id, source_clean = source.split('\t')
            assert predict_id[2:] == ground_id [2:]
            assert ground_id[2:] == source_id[2:]
        except Exception:
            print("An error occurred in source: {}".format(source))
            continue
        data.append((predict_clean, ground_clean, source_clean, predict_id[2:]))

    return data


def evaluate(data: List, target_delimiter: str):

    def evaluate_example(_predict_str: str, _ground_str: str):
        _predict_spans = _predict_str.split(target_delimiter)
        _ground_spans = _ground_str.split(target_delimiter)
        _predict_values = defaultdict(lambda: 0)
        _ground_values = defaultdict(lambda: 0)
        for span in _predict_spans:
            try:
                _predict_values[float(span)] += 1
            except ValueError:
                _predict_values[span.strip()] += 1
        for span in _ground_spans:
            try:
                _ground_values[float(span)] += 1
            except ValueError:
                _ground_values[span.strip()] += 1
        _is_correct = _predict_values == _ground_values
        return _is_correct

    correct_num = 0
    correct_arr = []
    total = len(data)

    for example in data:
        predict_str, ground_str, source_str, predict_id = example
        is_correct = evaluate_example(predict_str, ground_str)
        if is_correct:
            correct_num += 1
        correct_arr.append(is_correct)
    
    log_str = "Correct / Total : {} / {}, Denotation Accuracy : {:.3f}".format(correct_num, total, correct_num / total)
    # print("Correct / Total : {} / {}, Denotation Accuracy : {:.3f}".format(correct_num, total, correct_num / total))
    # write_all_result("Correct / Total : {} / {}, Denotation Accuracy : {:.3f}\n".format(correct_num, total, correct_num / total), )
    return correct_arr, log_str



def write_all_result(result_str, checkpoint_dir = None):
    print(checkpoint_dir + '/all_result.txt')
    fout = open("/users10/hanc/commonsenseGeneration/predict/" + checkpoint_dir + '/all_result.txt', 'a', encoding='utf8')
    fout.write(result_str + '\n')
    fout.close()




def evaluate_generate_file(generate_file_path, target_delimiter, name, checkpoint_dir = None):
    name = name.split('/')[-1].split('_')[-1]
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)
        correct_arr = None
        log_str = None

        correct_arr, log_str = evaluate(data, target_delimiter)


        # write into eval file
        # write_all_result(log_str, checkpoint_dir=checkpoint_dir)
        eval_file_path = generate_file_path +'_'+ name +".csv"
        eval_file = open(eval_file_path, "w", encoding="utf8")
        eval_file.write("Score\tPredict\tGolden\tSource\tID\n")
        for example, correct in zip(data, correct_arr):
            eval_file.write(str(correct) + "\t" + "\t".join(example) + "\n")
        eval_file.close()
