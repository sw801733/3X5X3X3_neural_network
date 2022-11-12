#define _CRT_SECURE_NO_WARNINGS 
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_INPUT 3
#define MAX_HIDDEN_1 5
#define MAX_HIDDEN_2 3
#define MAX_OUTPUT 3
#define LEARNING_DATA 210
#define LEARNING_RATE 0.5
#define MAX_TESTCASE 20
#define max(x,y) (x) > (y) ? (x) : (y)

double input[LEARNING_DATA][MAX_INPUT];
double target[LEARNING_DATA][MAX_OUTPUT];
double output[LEARNING_DATA][MAX_OUTPUT];
double total_error[LEARNING_DATA] = {0,};

double Xhidden_1[MAX_HIDDEN_1] = {0,}; 
double Ohidden_1[MAX_HIDDEN_1] = {0,};

double Xhidden_2[MAX_HIDDEN_2] = {0,}; 
double Ohidden_2[MAX_HIDDEN_2] = {0,}; 

double Xoutput[MAX_OUTPUT] = {0,}; 
double Ooutput[MAX_OUTPUT] = {0,}; 

double Winput_hidden1[MAX_INPUT][MAX_HIDDEN_1] = {{0.3, 0.2, 0.1, 0.5, 0.7},
                                                 {0.1, 0.9, 0.3, 0.4, 0.5},
                                                 {0.4, 0.5, 0.1, 0.6, 0.8}};

double Whidden1_hidden2[MAX_HIDDEN_1][MAX_HIDDEN_2] = {{0.7, 0.2, 0.3},
                                                      {0.3, 0.8, 0.4},
                                                      {0.9, 0.1, 0.6},
                                                      {0.4, 0.5, 0.2},
                                                      {0.5, 0.3, 0.9}};

double Whidden2_output[MAX_HIDDEN_2][MAX_OUTPUT] = {{0.2, 0.1, 0.7},
                                                   {0.3, 0.5, 0.4},
                                                   {0.9, 0.5, 0.2}};

double new_Winput_hidden1[MAX_INPUT][MAX_HIDDEN_1];
double new_Whidden1_hidden2[MAX_HIDDEN_1][MAX_HIDDEN_2];
double new_Whidden2_output[MAX_HIDDEN_2][MAX_OUTPUT];

// a = (O - target) * O * (1 - O)
double a_Ooutput[MAX_OUTPUT] = {0,};
double a_Ohidden_2[MAX_HIDDEN_2] = {0,};
double a_Ohidden_1[MAX_HIDDEN_1] = {0,};


void init_layer()
{
    for(int i = 0; i < MAX_HIDDEN_1; i++)
    {
        Xhidden_1[i] = 0;
        Ohidden_1[i] = 0;
    }
    
    for(int i = 0; i < MAX_HIDDEN_2; i++)
    {
        Xhidden_2[i] = 0;
        Ohidden_2[i] = 0;
    }
    
    for(int i = 0; i < MAX_OUTPUT; i++)
    {
        Xoutput[i] = 0;
        Ooutput[i] = 0;
    }

    for(int i = 0; i < LEARNING_DATA; i++)
    {
        total_error[i] = 0;
    }
}

void Forward_Propagation(int _idx,double _input[])
{
    // input to hidden_1
    for(int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_INPUT; j++)
        {
            Xhidden_1[i] += _input[j] * Winput_hidden1[j][i]; 
        }
        Ohidden_1[i] = 1 / (1 + exp(-Xhidden_1[i]));
    }

    // hidden_1 to hidden_2
    for(int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            Xhidden_2[i] += Ohidden_1[j] * Whidden1_hidden2[j][i]; 
        }
        Ohidden_2[i] = 1 / (1 + exp(-Xhidden_2[i]));
    }

    // hidden_2 to output
    for(int i = 0; i < MAX_OUTPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            Xoutput[i] += Ohidden_2[j] * Whidden2_output[j][i]; 
        }
        Ooutput[i] = 1 / (1 + exp(-Xoutput[i]));
        output[_idx][i] = Ooutput[i];
    }
}

void Update_Weight(FILE* Wh2_out, FILE* Wh1_h2, FILE* Win_h1)
{
    
    // Weight hidden2 to output
    for (int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_OUTPUT; j++)
        {
            fprintf(Wh2_out, "%f\n",Whidden2_output[i][j]);
            Whidden2_output[i][j] = new_Whidden2_output[i][j];
        }
    }

    // Weight hidden1 to hidden2
    for (int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            fprintf(Wh1_h2, "%f\n",Whidden1_hidden2[i][j]);
            Whidden1_hidden2[i][j] = new_Whidden1_hidden2[i][j];
        }
    }

    // Weight input to hidden1
    for (int i = 0; i < MAX_INPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            fprintf(Win_h1, "%f\n",Winput_hidden1[i][j]);
            Winput_hidden1[i][j] = new_Winput_hidden1[i][j];
        }
    }
}

void Back_Propagation(double _input[], double _target[])
{
    // output to hidden_2
    // a_Ooutput[i] = (Ooutput[i] - target[i]) * Oouput[i] * (1 - Ooutput[i])
    for(int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_OUTPUT; j++)
        {
            a_Ooutput[j] = (Ooutput[j] - _target[j]) * Ooutput[j] * (1 - Ooutput[j]); 
            new_Whidden2_output[i][j] = Whidden2_output[i][j] - LEARNING_RATE * a_Ooutput[j] * Ohidden_2[i];
        }
        
    }

    // hidden_2 to hidden_1
    // a_Ohidden_2[i] = 시그마j (cj * Whidden2_output[i][j]) * Ohidden_2[i] * (1 - Ohidden_2[i])
    for (int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            double temp = 0;
            for(int k = 0; k < MAX_OUTPUT; k++)
            {
                temp += a_Ooutput[k] * Whidden2_output[j][k];
            }
            a_Ohidden_2[j] = temp * Ohidden_2[j] * (1 - Ohidden_2[j]);
            new_Whidden1_hidden2[i][j] = Whidden1_hidden2[i][j] - LEARNING_RATE * a_Ohidden_2[j] * Ohidden_1[i];
        }
    }

    // hidden_1 to input
    // a_Ohidden_1[i] = 시그마j (a_Ohidden_2[j] * Winput_hidden1[i][j]) * Ohidden_1[i] * (1 - Ohidden_1[i])
    for (int i = 0; i < MAX_INPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            double temp = 0;
            for(int k = 0; k < MAX_HIDDEN_2; k++)
            {
                temp += a_Ohidden_2[k] * Whidden1_hidden2[j][k];
            }
            a_Ohidden_1[j] = temp * Ohidden_1[j] * (1 - Ohidden_1[j]);
            new_Winput_hidden1[i][j] = Winput_hidden1[i][j] - LEARNING_RATE * a_Ohidden_1[j] * _input[i];
        }
    }

}

void Cal_Total_Error(int _idx, double _target[][MAX_OUTPUT] ,double _output[][MAX_OUTPUT])
{
    for(int i = 0; i < MAX_OUTPUT; i++)
        total_error[_idx] += (_target[_idx][i] - _output[_idx][i]) * (_target[_idx][i] - _output[_idx][i]) / 2;
}


double* make_dataset(double _target)
{
    
    double x, y, z, r;
    double origin = 1; // 원점
    static double _dataset[4];
    double x_range, y_range, z_range;

    // x y 가 같아야 함
    while(1)
    {
        if (_target == 1.0)
        {
            // x : 1 ~ 2 | y : 1 ~ 2 | z : 0 ~ 1
            x_range = (rand() / (double)RAND_MAX);     // 0 ~ 1
            y_range = (rand() / (double)RAND_MAX);     // 0 ~ 1
            z_range = (rand() / (double)RAND_MAX);     // 0 ~ 1
            r = 1;
        }
        else if (_target == 2.0)
        {
            // x : 1 ~ 1.5 | y : 0.5 ~ 1 | z : 1.5 ~ 2.5
            x_range = (rand() / (double)RAND_MAX) / 2;   // 0 ~ 0.5
            y_range = -(rand() / (double)RAND_MAX) / 2;  // 0 ~ 0.5
            z_range = 1.5 + (rand() / (double)RAND_MAX);   // 1 ~ 2
            r = 0.5;
        }
        else if (_target == 3.0)
        {
            // x : 0.2 ~ 1 | y : 1 ~ 1.8 | z : 3 ~ 4
            x_range = -(rand() / (double)RAND_MAX) * (0.8);     // -1 ~ 0 
            y_range = (rand() / (double)RAND_MAX) * (0.8);      // 0 ~ 1
            z_range = 3 + (rand() / (double)RAND_MAX);  // 2.0 ~ 3.0
            r = 0.8;
        }

        x = origin + x_range;
        y = origin + y_range;
        z = z_range;

        // 부채꼴 반지름과 (x,y) 거리
        if (((x - origin) * (x - origin)) + ((y - origin) * (y - origin)) <= (r * r))
        {
            _dataset[0] = x;
            _dataset[1] = y;
            _dataset[2] = z;
            _dataset[3] = _target;

            break;
        }
    }
    return _dataset;

}

int main()
{   
    FILE* error_txt = fopen("error.txt", "w");
    FILE* testcase_xyz = fopen("testcase_xyz.txt", "w");
    FILE* testcase_output = fopen("testcase_output.txt", "w");
    FILE* dataset = fopen("dataset.txt", "w");

    FILE* Wh2_out = fopen("Wh2_out.txt", "w");
    FILE* Wh1_h2 = fopen("Wh1_h2.txt", "w");
    FILE* Win_h1 = fopen("Win_h1.txt", "w");

    double _target = 1.0;
    for(int i = 0; i < LEARNING_DATA; i++)
    {
        double* temp = make_dataset(_target);
        input[i][0] = temp[0];
        input[i][1] = temp[1];
        input[i][2] = temp[2];
        
        if (temp[3] == 1.0)
        {
            target[i][0] = 1;
            target[i][1] = 0;
            target[i][2] = 0;
            _target = 2.0;
        }
        if (temp[3] == 2.0)
        {
            target[i][0] = 0;
            target[i][1] = 1;
            target[i][2] = 0;
            _target = 3.0;
        }
        if (temp[3] == 3.0)
        {
            target[i][0] = 0;
            target[i][1] = 0;
            target[i][2] = 1;
            _target = 1.0;
        }
        fprintf(dataset, "%f %f %f\n", input[i][0],input[i][1],input[i][2]);
    }

    for(int k = 0; k < 5; k++)
    {
        for (int i = 0; i < LEARNING_DATA; i++)
        {
            Forward_Propagation(i, input[i]);
            
            Back_Propagation(input[i], target[i]);

            Update_Weight(Wh2_out, Wh1_h2, Win_h1);

            Cal_Total_Error(i, target, output);
            
            fprintf(error_txt, "%f\n", total_error[i]);
            init_layer();

        }
    }
    printf("%f %f %f", target[LEARNING_DATA - 1][0] - output[LEARNING_DATA - 1][0], target[LEARNING_DATA - 1][1] - output[LEARNING_DATA - 1][1], target[LEARNING_DATA - 1][2] - output[LEARNING_DATA - 1][2]);

    // test case
    double arr[MAX_TESTCASE][MAX_OUTPUT] = {{1.02, 1.05, 0.09},  // R, x 축에 가깝게
                                   {1.08, 0.56, 1.60},  // Y
                                   {0.96, 1.03, 3.11},  // B
                                   {1.50, 1.05, 0.97},   // R, y 축에 가깝게
                                   {1.46, 0.95, 2.46},  // Y
                                   {0.93, 1.11, 3.97},  // B
                                   {1.53, 1.56, 0.03},  // R, x,y 축은 구역 기준 중앙에 가깝게, z는 구역 기준 바닥에 가깝게 
                                   {1.23, 0.74, 1.56},  // Y
                                   {0.63, 1.49, 3.05},  // B
                                   {1.57, 1.53, 0.98},  // R, x,y 축은 구역 기준 중앙에 가깝게, z는 구역 기준 위쪽에
                                   {1.24, 0.77, 2.46},  // Y
                                   {0.66, 1.44, 3.95},  // B
                                   {1.58, 1.52, 0.55},  // R, x, y, z 축 모두 구역기준 중앙에 가깝게
                                   {1.25, 0.76, 2.03},  // Y
                                   {0.63, 1.47, 3.53},  // B
                                   {1.41, 0.88, 0.32},  // R 구역에서 y 축으로 벗어난 곳
                                   {1.62, 0.75, 1.82},  // Y 구역에서 x 축으로 벗어난 곳
                                   {0.43, 1.79, 4.21},  // B 구역에서 z 축으로 벗어난 곳
                                   {0.58, 1.63, 1.33},  // R 구역과 Y 구역 사이
                                   {0.31, 0.22, 2.79},  // Y 구역과 B 구역 사이
                                    };

    for(int i = 0; i < MAX_TESTCASE; i++)
    {
        char result;
        double _max = 0;
        Forward_Propagation(i, arr[i]);

        for(int j = 0; j < MAX_OUTPUT; j++)
        {
            _max = max(_max, output[i][j]);
            fprintf(testcase_xyz, "%f ", arr[i][j]);
            printf("%f ", output[i][j]);
        }
        if (_max == output[i][0])
        {
            result = 'R';
            fprintf(testcase_xyz, "%f %f %f\n", 1.0, 0.0, 0.0);
        }
        else if (_max == output[i][1])
        {
            result = 'Y';
            fprintf(testcase_xyz, "%f %f %f\n", 0.0, 1.0, 0.0);
        }
        else if (_max == output[i][2])
        {
            result = 'B';
            fprintf(testcase_xyz, "%f %f %f\n", 0.0, 0.0, 1.0);
        }
        printf("%c\n", result);
        fprintf(testcase_output, "%c\n", result);
        init_layer();
    }
    return 0;
}