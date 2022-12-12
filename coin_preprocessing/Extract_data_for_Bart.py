import json

def get_bart_training_corpus(json_file_name,step_number,target_file_path):
    with open(json_file_name,'r',encoding='utf-8') as file:
        Overall_dict=json.load(file)
    Overall_dict=Overall_dict['database']
    data_list=[]
    print(len(Overall_dict.keys()))
    for video_name in Overall_dict.keys():
        sample_dict={}
        sample_piece = Overall_dict[video_name]
        ID=sample_piece['recipe_type']
        Target_Label=sample_piece['class']
        annotations=sample_piece['annotation']
        Action_Len=len(annotations)
        number_of_samples = Action_Len - step_number + 1
        for j in range (number_of_samples):
            Action_Label=[]
            for i in range (j,j+step_number):
                Action_Label.append(annotations[i]['label'])
            sample_dict={'ID':ID,'Target_Label':Target_Label,'Action_Len':step_number,'Action_Label':Action_Label}
            data_list.append(sample_dict)
    with open(target_file_path,"w") as f:
        json.dump(data_list,f)

if __name__ == '__main__':
    get_bart_training_corpus(YOUR_ORIGINAL_JSON_FILE,STEP_NUMBER,YOUR_TARGET_JSON_FILE_FOR_LM)