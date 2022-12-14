#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_INPUT 3
#define MAX_HIDDEN_1 5
#define MAX_HIDDEN_2 3
#define MAX_OUTPUT 3
#define LEARNING_DATA 200


float input[LEARNING_DATA][MAX_INPUT];
float target[LEARNING_DATA][MAX_OUTPUT];
float output[LEARNING_DATA][MAX_OUTPUT];

float Xhidden_1[MAX_HIDDEN_1] = {0,}; 
float Ohidden_1[MAX_HIDDEN_1] = {0,};

float Xhidden_2[MAX_HIDDEN_2] = {0,}; 
float Ohidden_2[MAX_HIDDEN_2] = {0,}; 

float Xoutput[MAX_OUTPUT] = {0,}; 
float Ooutput[MAX_OUTPUT] = {0,}; 

float Winput_hidden1[MAX_INPUT][MAX_HIDDEN_1] = {{0.3, 0.4, 0.5, 0.3, 0.4},
                                                 {0.7, 0.3, 0.4, 0.2, 0.5},
                                                 {0.5, 0.8, 0.8, 0.8, 0.4}};

float Whidden1_hidden2[MAX_HIDDEN_1][MAX_HIDDEN_2] = {{0.4, 0.1, 0.6},
                                                      {0.2, 0.8, 0.4},
                                                      {0.6, 0.5, 0.7},
                                                      {0.4, 0.3, 0.2},
                                                      {0.5, 0.4, 0.5}};

float Whidden2_output[MAX_HIDDEN_2][MAX_OUTPUT] = {{0.5, 0.7, 0.3},
                                                   {0.2, 0.6, 0.5},
                                                   {0.3, 0.8, 0.4}};

float new_Winput_hidden1[MAX_INPUT][MAX_HIDDEN_1];
float new_Whidden1_hidden2[MAX_HIDDEN_1][MAX_HIDDEN_2];
float new_Whidden2_output[MAX_HIDDEN_2][MAX_OUTPUT];

// a = (O - target) * O * (1 - O)
float a_Ooutput[MAX_OUTPUT] = {0,};
float a_Ohidden_2[MAX_HIDDEN_2] = {0,};
float a_Ohidden_1[MAX_HIDDEN_1] = {0,};

// Learning rate
float alpha = 0.5;

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
}

void Forward_Propagation(int _idx,float _input[])
{
    ///printf("%f\n", _input[0]);
    // input to hidden_1
    for(int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_INPUT; j++)
        {
            Xhidden_1[i] += _input[j] * Winput_hidden1[j][i]; 
            //printf("%f ", Winput_hidden1[j][i]);
        }
        //printf("\n");
        Ohidden_1[i] = 1 / (1 + exp(-Xhidden_1[i]));
    }

    //printf("\n");
    // hidden_1 to hidden_2
    for(int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            Xhidden_2[i] += Ohidden_1[j] * Whidden1_hidden2[j][i]; 
            //printf("%f ", Whidden1_hidden2[j][i]);
        }
        //printf("\n");
        Ohidden_2[i] = 1 / (1 + exp(-Xhidden_2[i]));
    }
    //printf("\n");
    // hidden_2 to output
    for(int i = 0; i < MAX_OUTPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            Xoutput[i] += Ohidden_2[j] * Whidden2_output[j][i]; 
            //printf("%f ", Whidden2_output[j][i]);
        }
        //printf("\n");
        Ooutput[i] = 1 / (1 + exp(-Xoutput[i]));
        output[_idx][i] = Ooutput[i];
    }
    //printf("\n");
    
}

void Update_Weight()
{
    for (int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_OUTPUT; j++)
        {

            Whidden2_output[i][j] = new_Whidden2_output[i][j];
        }
    }

    for (int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            Whidden1_hidden2[i][j] = new_Whidden1_hidden2[i][j];

        }
    }

    for (int i = 0; i < MAX_INPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            Winput_hidden1[i][j] = new_Winput_hidden1[i][j];

        }
    }
}

void Back_Propagation(float _input[], float _target[])
{
    // output to hidden_2
    // a_Ooutput[i] = (Ooutput[i] - target[i]) * Oouput[i] * (1 - Ooutput[i])
    for(int i = 0; i < MAX_HIDDEN_2; i++)
    {
        for(int j = 0; j < MAX_OUTPUT; j++)
        {
            a_Ooutput[j] = (Ooutput[j] - _target[j]) * Ooutput[j] * (1 - Ooutput[j]); 
            new_Whidden2_output[i][j] = Whidden2_output[i][j] - alpha * a_Ooutput[j] * Ohidden_2[i];
        }
        
    }

    // hidden_2 to hidden_1
    // a_Ohidden_2[i] = ?????????j (cj * Whidden2_output[i][j]) * Ohidden_2[i] * (1 - Ohidden_2[i])
    for (int i = 0; i < MAX_HIDDEN_1; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_2; j++)
        {
            float temp = 0;
            for(int k = 0; k < MAX_OUTPUT; k++)
            {
                temp += a_Ooutput[k] * Whidden2_output[j][k];
            }
            a_Ohidden_2[j] = temp * Ohidden_2[j] * (1 - Ohidden_2[j]);
            new_Whidden1_hidden2[i][j] = Whidden1_hidden2[i][j] - alpha * a_Ohidden_2[j] * Ohidden_1[i];
        }
    }

    // hidden_1 to input
    // a_Ohidden_1[i] = ?????????j (a_Ohidden_2[j] * Winput_hidden1[i][j]) * Ohidden_1[i] * (1 - Ohidden_1[i])
    for (int i = 0; i < MAX_INPUT; i++)
    {
        for(int j = 0; j < MAX_HIDDEN_1; j++)
        {
            float temp = 0;
            for(int k = 0; k < MAX_HIDDEN_2; k++)
            {
                temp += a_Ohidden_2[k] * Whidden1_hidden2[j][k];
            }
            a_Ohidden_1[j] = temp * Ohidden_1[j] * (1 - Ohidden_1[j]);
            new_Winput_hidden1[i][j] = Winput_hidden1[i][j] - alpha * a_Ohidden_1[j] * _input[i];
        }
    }
    Update_Weight();

}

int main()
{   
    float x,y,z;
    float a,b,c;
    a = 0.5;
    b = 0.0;
    c = 0.0;
    for(int i = 0; i < LEARNING_DATA; i++)
    {
        // if (i == 80)
        // {
        //     a = 0.0;
        //     b = 0.5;
        //     c = 0.0;
        // }
        // else if (i == 160)
        // {
        //     a = 0.0;
        //     b = 0.0;
        //     c = 0.5;
        // }

        // x = a + (rand() / (float) RAND_MAX/2);
        // y = b + (rand() / (float) RAND_MAX/2);
        // z = c + (rand() / (float) RAND_MAX/2);
        // input[i][0] = x;
        // input[i][1] = y;
        // input[i][2] = z;
        // target[i][0] = a * 2;
        // target[i][1] = b * 2;
        // target[i][2] = c * 2;
        scanf("%f %f %f %f %f %f", &input[i][0], &input[i][1], &input[i][2], &target[i][0], &target[i][1], &target[i][2]);
        
    }



    for(int k = 0; k < 10; k++)
    {
        for (int i = 0; i < LEARNING_DATA; i++)
        {
            

            // printf("Error : ");
            // for(int j = 0; j < 3; j++)
            //     printf("%f ", target[i][j] - output[i][j]);

            Forward_Propagation(i, input[i]);

            Back_Propagation(input[i], target[i]);

            init_layer();
            
            
            // printf("\n");
            //init_layer();
            // // Forward_Propagation(i, input[i]);
            
            // printf("After  Learning : ");
            // for(int j = 0; j < 3; j++)
            //     printf("%f ", output[i][j]);
            // printf("\n");
            // printf("==============================================================\n");
            
        }
    }
    float arr[1][3] = {{0.1, 0.9, 0.9}};

    Forward_Propagation(0, arr[0]);
    for(int j = 0; j < 3; j++)
        printf("%f ", output[0][j]);
    // init_layer();
    // float arr[1][3] = {{0.1, 0.8, 0.1}};
    // Forward_Propagation(0, arr[0]);
    // // printf("Error : ");
    // for(int j = 0; j < 3; j++)
    //     printf("%f ", output[0][j]);
    // ????????? ??????   
    // // output to hidden2
    // float temp1 = (Ooutput[0] - target[0][0]) * Ooutput[0] * (1 - Ooutput[0]);
    // printf("%f\n", temp1);
    // printf("%f\n", a_Ooutput[0]);

    // // hidden2 to hidden1
    // float temp2 = (a_Ooutput[0] * Whidden2_output[1][0] + a_Ooutput[1] * Whidden2_output[1][1] + a_Ooutput[2] * Whidden2_output[1][2]) * Ohidden_2[1] * (1 - Ohidden_2[1]);
    // printf("%f\n", temp2);
    // printf("%f\n", a_Ohidden_2[1]);

    // // hidden1 to input
    // float temp3 = (a_Ohidden_2[0] * Whidden1_hidden2[0][0] + a_Ohidden_2[1] * Whidden1_hidden2[0][1] + a_Ohidden_2[2] * Whidden1_hidden2[0][2]) * Ohidden_1[0] * (1 - Ohidden_1[0]);
    // printf("%f\n", temp3);
    // printf("%f", a_Ohidden_1[0]);
    

    return 0;
}