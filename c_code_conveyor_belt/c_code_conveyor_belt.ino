#include <math.h>

// Select the ML model to use
#define EXTRA_TREES_10 0
#define EXTRA_TREES_25 0
#define XGB_10  0
#define XGB_25  1
#define RF_10  0
#define RF_25  0
#define SUPPORT_VM 0
#define NAIVE_BAYES 0
#define MLP_12 0
#define MLP_5 0
#define DT 0


#define MEDIAN_FILTER_ORDER 5


#if EXTRA_TREES_10
  #include "xtree10_model.h"
#elif EXTRA_TREES_25
  #include "xtree25_model.h"
#elif XGB_10
  #include "xgb10_model.h"
    Eloquent::ML::Port::XGBClassifier clf;
#elif XGB_25
  #include "xgb25_model.h"
    Eloquent::ML::Port::XGBClassifier clf;
#elif RF_10
  #include "rf10_model.h"
  Eloquent::ML::Port::RandomForest clf;
#elif RF_25
  #include "rf25_model.h"
  Eloquent::ML::Port::RandomForest clf;
#elif SUPPORT_VM
  #include "svm_model.h"
  Eloquent::ML::Port::SVM clf;
#elif NAIVE_BAYES
  #include "nb_model.h"
#elif MLP_12
  #include "mlp_12neurons_model.h"
#elif MLP_5
  #include "mlp_5neurons_model.h"
#elif DT
  #include "dt_model.h"
  Eloquent::ML::Port::DecisionTree clf;
#endif



#define N_SAMPLES 115   //total number of input instances,
#define N_FEATURES 12   
#define N_INPUTS 3      //number of inputs variables
#define REPEAT_EXP 1000  //number of times to repeat the experiments


const float v_input[N_SAMPLES][N_INPUTS]={
   {34.38,26.16,0.0},
   {34.43,26.15,0.0},
   {34.4,26.13,0.0},
   {35.97,23.77,3.98},
   {44.08,24.31,30.05},
   {44.57,25.24,43.97},
   {44.26,25.19,45.18},
   {43.51,25.1,45.23},
   {43.36,25.0,45.22},
   {43.07,24.96,45.22},
   {43.01,24.92,45.21},
   {42.91,24.88,45.24},
   {42.99,24.85,45.21},
   {42.84,24.81,45.21},
   {43.18,24.78,45.21},
   {43.15,24.75,45.23},
   {43.03,24.73,45.28},
   {42.96,24.69,45.23},
   {43.01,24.66,45.23},
   {43.03,24.64,45.21},
   {43.01,24.61,45.21},
   {42.83,24.59,45.21},
   {43.94,24.59,45.25},
   {42.84,24.55,45.22},
   {43.04,24.54,45.23},
   {42.15,24.49,45.24},
   {45.0,24.51,45.21},
   {47.38,24.54,45.21},
   {49.51,24.52,45.19},
   {63.38,24.56,45.14},
   {63.01,24.58,45.18},
   {44.88,24.56,45.35},
   {45.71,24.56,45.25},
   {52.44,24.58,45.15},
   {60.1,24.59,45.12},
   {81.59,24.55,45.01},
   {120.45,24.43,44.57},
   {127.08,24.43,44.54},
   {126.83,24.41,44.55},
   {126.88,24.39,44.54},
   {127.9,24.38,44.49},
   {129.62,24.35,44.44},
   {130.65,24.34,44.44},
   {131.66,24.33,44.42},
   {132.28,24.33,44.41},
   {132.4,24.31,44.34},
   {133.08,24.29,44.38},
   {132.99,24.28,44.38},
   {132.04,24.26,44.37},
   {130.29,24.27,44.39},
   {128.57,24.25,44.37},
   {128.31,24.25,44.37},
   {128.33,24.23,44.35},
   {128.56,24.26,44.34},
   {128.26,24.25,44.34},
   {127.95,24.23,44.36},
   {127.47,24.19,44.34},
   {127.28,24.21,44.34},
   {127.11,24.2,44.32},
   {127.02,24.19,44.34},
   {127.16,24.2,44.37},
   {127.05,24.16,44.31},
   {127.2,24.14,44.28},
   {127.17,24.14,44.29},
   {127.17,24.14,44.29},
   {127.28,24.14,44.3},
   {127.48,24.12,44.25},
   {127.54,24.13,44.29},
   {127.68,24.15,44.37},
   {127.49,24.13,44.34},
   {127.04,24.11,44.34},
   {126.88,24.12,44.32},
   {126.6,24.14,44.3},
   {126.31,24.14,44.3},
   {126.15,24.11,44.21},
   {125.83,24.11,44.35},
   {125.21,24.1,44.34},
   {125.04,24.09,44.31},
   {124.66,24.06,44.23},
   {123.21,24.09,44.24},
   {121.07,24.09,44.31},
   {119.04,24.09,44.33},
   {110.15,24.18,44.5},
   {106.75,24.19,44.46},
   {112.29,24.16,44.4},
   {111.72,24.16,44.44},
   {111.34,24.12,44.38},
   {111.68,24.14,44.38},
   {110.99,24.17,44.42},
   {111.23,24.14,44.39},
   {111.78,24.13,44.35},
   {114.56,24.13,44.31},
   {116.56,24.1,44.25},
   {113.24,24.09,44.27},
   {107.6,24.14,44.32},
   {94.87,24.24,44.41},
   {58.06,24.39,44.78},
   {41.67,24.28,45.05},
   {41.24,24.23,44.97},
   {41.3,24.25,44.94},
   {41.38,24.27,44.94},
   {41.42,24.27,44.93},
   {41.48,24.31,45.02},
   {41.42,24.32,44.95},
   {41.4,24.33,44.93},
   {41.44,24.35,44.94},
   {41.46,24.35,44.91},
   {41.56,24.38,44.91},
   {41.69,24.39,44.88},
   {38.8,23.92,41.37},
   {30.01,24.08,7.33},
   {31.32,25.23,0.0},
   {31.35,25.39,0.0},
   {31.58,25.54,0.0},
   {32.0,25.66,0.0}
   };

// const float v_input[N_SAMPLES][N_INPUTS]={
//     {32.0,25.66,0.0},
//     {32.1,25.67,0.1},
//     {32.2,25.68,0.2},
//     {32.3,25.69,0.3},
//     {32.4,25.65,0.4}
//     };
  
// Function to swap two elements in an array
void swap(unsigned int *x, unsigned int *y) {
    float temp = *x;
    *x = *y;
    *y = temp;
}

// Function to perform bubble sort on the array
void bubbleSort(unsigned int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}    

// Function to compute the median of a 1D array of 5 elements
unsigned int computeMedian(unsigned int arr[], int n) {
    // Sort the array
    bubbleSort(arr, n);

    // Calculate the median
    return arr[n / 2];
}


float v_features[N_FEATURES]={0};
unsigned int median_filter_buffer[MEDIAN_FILTER_ORDER]={0};
unsigned int inference_time[REPEAT_EXP] = {};
unsigned int inference_index = 0;
unsigned int min_inference_time = 0xFFFFFFFF;
unsigned int max_inference_time = 0;
float mean_inference_time = 0;
float std_inference_time = 0;
unsigned int output_state;

void CalculateInferenceStandardDeviation()
{
  unsigned int sum = 0;
  float mean = 0;
  float std = 0;
  for (int i = 0; i < REPEAT_EXP; i++)
  {
    sum += inference_time[i];
  }
  mean = float(sum) / REPEAT_EXP;
  mean_inference_time = mean;
  for (int i = 0; i < REPEAT_EXP; i++)
  {
    std += (inference_time[i] - mean) * (inference_time[i] - mean);
  }
  std = std / REPEAT_EXP;
  std_inference_time = sqrt(std);
}

void CalculateMaxMinInferenceTime()
{
  for (int i = 0; i < REPEAT_EXP; i++)
  {
    if (inference_time[i] < min_inference_time)
    {
      min_inference_time = inference_time[i];
    }
    if (inference_time[i] > max_inference_time)
    {
      max_inference_time = inference_time[i];
    }
  }
}


void compute_features(float high_pres_m2,float high_pres_m1,float high_pres,float high_pres_p1,float high_pres_p2,float low_pres_m2,float low_pres_m1,
        float low_pres,float low_pres_p1,float low_pres_p2,float speed_m2,float speed_m1,float speed,float speed_p1,float speed_p2,float *v_features)
{
    v_features[0]=high_pres;
    v_features[1]=low_pres;
    v_features[2]=speed;
    v_features[3]=(speed_m1+speed+speed_p1)/3;  //mean 3
    v_features[4]=(high_pres_m1+high_pres+high_pres_p1)/3;  //mean 3
    v_features[5]=(low_pres_m1+low_pres+low_pres_p1)/3;  //mean 3
    v_features[6]=v_features[4]-v_features[5];  //difference order 3
    v_features[7]=(speed_m2+speed_m1+speed+speed_p1+speed_p2)/5;  //mean 5
    v_features[8]=(high_pres_m2+high_pres_m1+high_pres+high_pres_p1+high_pres_p2)/5;  //mean 5
    v_features[9]=(low_pres_m2+low_pres_m1+low_pres+low_pres_p1+low_pres_p2)/5;  //mean 5
    v_features[10]=v_features[8]-v_features[9];
    v_features[11]=high_pres-low_pres;          
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  unsigned int t1 = micros();

  for (int i=2; i< N_SAMPLES-2; i++){

//FEATURE EXTRACTION
    float high_pres_m2 = v_input[i-2][0];
    float high_pres_m1 = v_input[i-1][0];
    float high_pres = v_input[i][0];
    float high_pres_p1 = v_input[i+1][0];
    float high_pres_p2 = v_input[i+2][0];

    float low_pres_m2 = v_input[i-2][1];
    float low_pres_m1 = v_input[i-1][1];
    float low_pres = v_input[i][1];
    float low_pres_p1 = v_input[i+1][1];
    float low_pres_p2 = v_input[i+2][1];

    float speed_m2 = v_input[i-2][2];
    float speed_m1 = v_input[i-1][2];
    float speed = v_input[i][2];
    float speed_p1 = v_input[i+1][2];
    float speed_p2 = v_input[i+2][2];

    compute_features(high_pres_m2,high_pres_m1,high_pres,high_pres_p1,high_pres_p2,low_pres_m2,low_pres_m1,low_pres,low_pres_p1,low_pres_p2,
            speed_m2,speed_m1,speed,speed_p1,speed_p2,v_features);
    
//    for (int j = 0; j < N_FEATURES; j++) {
//      Serial.print(v_features[j]);
//      Serial.print(" - ");
//    }
//    Serial.println("");

//   ML MODEL
#if (NAIVE_BAYES || MLP_5 || MLP_12 ||  EXTRA_TREES_10 || EXTRA_TREES_25)
    volatile unsigned int pred = (unsigned int) model_predict(v_features, N_FEATURES);
#elif (DT || RF_10 || RF_25 || XGB_10|| XGB_25 ||  SUPPORT_VM)
    volatile unsigned int pred = (unsigned int) clf.predict(v_features);
#else
    Serial.print("select one classifier");
#endif

// MEDIAN FILTER

if (MEDIAN_FILTER_ORDER > 1){
  
  //circular buffer
  for (int j = 0; j < MEDIAN_FILTER_ORDER-1; j++) {
    median_filter_buffer[j]=median_filter_buffer[j+1];
  }
  median_filter_buffer[MEDIAN_FILTER_ORDER-1]=pred;

  // the actual output_state value is delayed 4 samples regarding the actual input (output of the ML classifier).
  // This happen because to compute the median value of a sample at time 'n', it required have the sample from n-3 up to n+3.
  output_state = computeMedian(median_filter_buffer, MEDIAN_FILTER_ORDER);
}

else{
  output_state = pred;
}



}

  unsigned int t2 = micros();

  if (inference_index != REPEAT_EXP)
  {
    inference_time[inference_index] = (t2 - t1)/(N_SAMPLES-4);
    inference_index++;
  }
  else
  {
    CalculateInferenceStandardDeviation();
    CalculateMaxMinInferenceTime();
    Serial.print("Average inference time : ");
    Serial.print(mean_inference_time);
    Serial.println(" microseconds");
    Serial.print("STD of inference time  : ");
    Serial.print(std_inference_time);
    Serial.println(" microseconds");
    Serial.print("Max inference time     : ");
    Serial.print(max_inference_time);
    Serial.println(" microseconds");
    Serial.print("Min inference time     : ");
    Serial.print(min_inference_time);
    Serial.println(" microseconds");
//    Serial.print("Prediction ML (actual) : ");
//    Serial.println(pred);
//    Serial.print("Prediction ML (actual-3) : ");
//    Serial.println(pred);
//    Serial.print("Output median filter     : ");
//    Serial.println(output_state);
    Serial.println("--------------------------------------------------");
    inference_index = 0;
    max_inference_time = 0;
    min_inference_time = 0xFFFFFFFF;
    mean_inference_time = 0;
    std_inference_time = 0;
  }
  
}
