
# from detectionofballotpaper.objectdetect.objdetecteval.metrics import image_metrics as im, coco_metrics as cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from objectdetect.objdetecteval.metrics import image_metrics as im, coco_metrics as cm
import os

class Metrics():

    def get_image_id(self, filename):
           
        return int(filename.split('_')[1].split('.')[0]) - 1
    
    def metrics(self, prediction_labels, true_labels_path, label_to_id, files_path):
        pred_label = []
        true_labels = pd.read_csv(true_labels_path)
        true_labels = true_labels.rename(columns={'image_id':'image_name','x1':'xmin','y1':'ymin','x2':'xmax','y2':'ymax'})
        
        # for data in prediction_labels:
        #     id = data['imageid'].cpu().numpy() # Use .item() to get the value as a standard Python scalar
        #     boxes = data['boxes'].cpu().numpy()
        #     labels = data['labels'].cpu().numpy()
        #     scores = data['scores'].cpu().numpy()
        #     image_name = data['imagename']

        #     for box, label, score in zip(boxes, labels, scores):
        #         xmin, ymin, xmax, ymax = box  # Unpack each box's coordinates
        #         pred_label.append({
        #             'image_id': id.item(),
        #             'xmin': xmin,
        #             'ymin': ymin,
        #             'xmax': xmax,
        #             'ymax': ymax,
        #             'label': label,  # Convert to Python scalar if necessary
        #             'score': score,  # Convert to Python scalar if necessary
        #             'image_name': image_name
        #         })

        # inv_label = {v:k for k,v in label_to_id.items()}            
        # preds_df = pd.DataFrame(pred_label)
        # preds_df['category_id'] = preds_df['label']
        # preds_df['label'] = preds_df['label'].replace(inv_label)  
        
        
        # preds_df.to_excel(os.path.join(files_path, 'preds_df.xlsx'))


        # true_labels_df = pd.DataFrame(true_labels)
        
        # true_labels_df['category_id'] = true_labels_df['label'].replace(label_to_id) 
        # true_labels_df['image_id'] = true_labels_df['image_name'].apply(self.get_image_id)

        # true_labels_df.to_excel(os.path.join(files_path, 'true_labels_df.xlsx'))        
        
        # pred = pd.read_excel(os.path.join(files_path, 'preds_df.xlsx'))
        # predd = pd.DataFrame(pred)
        # true = pd.read_excel(os.path.join(files_path, 'true_labels_df.xlsx'))
        # true = pd.DataFrame(true)

        # infer_df = im.get_inference_metrics_from_df(predd, trued)
        # infer_df.to_excel(os.path.join(files_path, 'infer_df.xlsx'))
        class_summary = pd.read_excel('./artifacts/faster_rcnn_files/total_comparisons_normalized.xlsx')

        class_summary_df = im.summarise_inference_metrics(class_summary)
        class_summary_df.to_excel(os.path.join(files_path, 'class_summary_df.xlsx'))
   
    
    def generate_separate_precision_recall_curves(self, df):
        unique_classes = df['class'].unique()
        n_classes = 43
        
        # Calculate the number of rows/columns for the subplot grid
        n_cols = 5  # For example, 3 columns
        n_rows = (n_classes + n_cols - 1) // n_cols  # Calculate rows needed
        
        plt.figure(figsize=(2 * n_cols, 3 * n_rows))  # Adjust figure size as needed
        
        # Loop through each class to plot its precision-recall curve
        for index, class_name in enumerate(sorted(unique_classes), start=1):
            plt.subplot(n_rows, n_cols, index)
            
            class_df = df[df['class'] == class_name].copy()
            class_df.sort_values('Confidence', ascending=False, inplace=True)
            
            # Initialize lists to store precision and recall values
            precisions = []
            recalls = []

            # Cumulatively calculate TP, FP, FN to determine precision and recall at each threshold
            cumulative_tp = 0
            cumulative_fp = 0
            for i in range(len(class_df)):
                cumulative_tp += class_df.iloc[i]['TP']
                cumulative_fp += class_df.iloc[i]['FP']
                cumulative_fn = class_df['FN'].sum()  # FN remains constant for a given class

                precision = cumulative_tp / (cumulative_tp + cumulative_fp) if (cumulative_tp + cumulative_fp) > 0 else 0
                recall = cumulative_tp / (cumulative_tp + cumulative_fn) if (cumulative_tp + cumulative_fn) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

            # Plotting for the current class
            # plt.plot(recalls, precisions, marker='.', label=f'{class_name}')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title(f'Class: {class_name}')
            # plt.legend()

            plt.plot(recalls, precisions, marker='.',color='pink', markerfacecolor='blue', label=f'{class_name}')
            plt.xlabel('Recall', fontsize=8)  # Adjust font size here
            plt.ylabel('Precision', fontsize=8)  # Adjust font size here
            # plt.title(f'Class: {class_name}', fontsize=8)  # Adjust font size here
            plt.legend(prop={'size': 5})  # Adjust legend font size here
            plt.tick_params(axis='both', which='major', labelsize=5) 
        # plt.xticks(rotation=45) 
        plt.tight_layout()
        # plt.savefig(f'../../outputdir/FasterRCNNMetrics.png', bbox_inches='tight', pad_inches=0, dpi=300)  
        # plt.subplots_adjust(bottom=0.3) # Adjust subplots to fit in the figure area
        # plt.show()


    def call_metrics(self, file_path):
        
        f1_micro_data = pd.read_excel(os.path.join(file_path, 'total_comparisons_normalized.xlsx')) 
        df = pd.DataFrame(f1_micro_data)     

        ##-----------------Precision Recall Curve Visualization
        # # obje.generate_precision_recall_curve(df, 'heart')
        # self.generate_separate_precision_recall_curves(df)

        # =CONCAT("\hline ",TEXT(B2,"0.00")," & ",TEXT(G2,"0.00")," & ", TEXT(H2,"0.00")," & ",TEXT(I2,"0.00")," & ",TEXT(J2,"0.00")," \\")


        # # -------------------------------Calculate Macro F1 Score
        data_summary = pd.read_excel(os.path.join(file_path, 'class_summary_df.xlsx'))   
        df_summary = pd.DataFrame(data_summary)
        df_summary['F1'] = 2 * (df_summary['Precision'] * df_summary['Recall']) / (df_summary['Precision'] + df_summary['Recall'])

        macro_f1_score = df_summary['F1'].mean()
        print("Macro F1 Score:", macro_f1_score)


        # #-------------------------------Calculating micro F1 Score
        total_TP = df['TP'].sum()
        total_FP = df['FP'].sum()
        total_FN = df['FN'].sum()

        precision_micro = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall_micro = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

        print(f"Micro F1 Score: {f1_micro}")
        print(f"Micro-average Precision: {precision_micro}")
        print(f"Micro-average Recall: {recall_micro}")

        return macro_f1_score, f1_micro, precision_micro, recall_micro
       


