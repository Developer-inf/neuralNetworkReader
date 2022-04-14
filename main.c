#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <string.h>

#define MAXFNAME 64
#define LINELENGTH 1024

char fn[MAXFNAME] = "keras_weights.json";	// file name of neural network structure
char inputsName[MAXFNAME];				// file name of inputs
FILE *rfd;								// read file descriptor

// states of program
enum States {
	findArr,	// find array to read layer
	readRow		// read row
};

// 2d array of weights with rows and columns
typedef struct Layer
{
	float **neurons;
	int r, c;
	float *wsums;
} Layer;

// function of activation
float f_activ(float val)
{
	return (val >= 0) * val;
}

// weight sum of neuron
float w_sum(float *inputs, Layer *l, int c)
{
	float sum = 0.f;
	for (int i = 0; i < l->r; i++)
	{
		sum += inputs[i] * l->neurons[i][c];
	}
	return sum;
}

// dot product of input neurons and weights to output neurons
void dot_prod(float *in, Layer *l)
{
	for (int i = 0; i < l->c; i++)
	{
		l->wsums[i] = f_activ(w_sum(in, l, i));
	}
}

// read inputs from file
int readInputs(float *in, int c)
{
	char buf[LINELENGTH];	// buffer
	int i = 0, j = 0, p = 0;// i, j, position
	int end = 0;			// is end of reading
	int len;				// string length in buffer
	float num;				// read number

	while (fgets(buf, LINELENGTH, rfd))
	{
		len = strlen(buf);
		i = 0;
		// finding start of reading
		while (buf[i] == ' ' || buf[i] == '[')
			i++;
		
		// while not end of reading and position in bounds
		while (i < len && !end)
		{
			// if new line read new line
			if (buf[i] == '\n')
				break;

			j = i;

			// reading number
			while (buf[i] != ',' && buf[i] != ']')
				i++;

			// if was closing bracket then we at the end
			if (buf[i] == ']')
				end = 1;

			// if position greater than count of inputs
			if (p >= c)
			{
				fprintf(stderr, "Too many inputs\n");
				exit(5);
			}

			// convert and store number
			buf[i++] = 0;
			num = atof(buf + j);
			in[p++] = num;
		}

		// return count of read numbers
		if (end)
			return p;
	}
	return p;
}

// read weights from file
Layer *readWeights(Layer *l, int *lcnt, int *inCnt)
{
	Layer *lastLayer;			// Layer for better understanding
	char buf[LINELENGTH];		// buffer
	char *arrayTemp = "array([";// template
	int rowOpen = 0;			// are we reading row
	int i, j;					// i, j
	enum States state = findArr;// prog state
	int len;					// len of string in buffer
	float num;					//read number

	*lcnt = 0;
	l = (Layer *)malloc(1 * sizeof(Layer));

	while (fgets(buf, LINELENGTH, rfd))
	{
		i = 0;
		len = strlen(buf);

		// choose actions according to state
		switch (state)
		{
		case findArr:
			// while not end of line
			while (i < len)
			{
				// if template can't fit continue
				if (i + 7 >= len)
					break;
				// if found template check
				if (strncmp(buf + i, arrayTemp, 7) == 0)
				{
					i += 7;
					// if it's array of weights change state
					if (buf[i] == '[')
					{
						rowOpen = 1;		// reading the row
						*lcnt += 1;			// increase count of layers
						l = (Layer *)realloc(l, *lcnt * sizeof(Layer));
						lastLayer = &l[*lcnt - 1];
						lastLayer->r = 1;	// increase rows
						lastLayer->c = 0;
						//allocating memory
						lastLayer->neurons = (float **)malloc(lastLayer->r * sizeof(float *));
						lastLayer->neurons[lastLayer->c] = (float *)malloc(sizeof(float));
						state = readRow;	// changing state
						i++;
						goto READROW;	// no need to read new line so go to next state and read row
					}
				}
				i++;
			}
			break;

		case readRow:
		READROW:
			// reading numbers
			while (i < len)
			{
				if (buf[i] == '\n')
					break;

				// finding row
				while (!rowOpen)
				{
					if (buf[i] == '[')
						rowOpen = 1;
					if (buf[i] == '\n')
						goto BREAK;
					i++;
				}

				// read number
				j = i;
				while (buf[i] != ',' && buf[i] != ']')
					i++;
				
				// if ']' then stop read row
				if (buf[i] == ']')
					rowOpen = 0;

				// convert and store value
				buf[i++] = 0;
				num = atof(buf + j);
				lastLayer->c++;
				lastLayer->neurons[lastLayer->r - 1] = (float *)realloc(lastLayer->neurons[lastLayer->r - 1], lastLayer->c * sizeof(float));
				lastLayer->neurons[lastLayer->r - 1][lastLayer->c - 1] = num;

				// if another ']' then stop read layer
				if (buf[i] == ']')
				{
					state = findArr;
					goto BREAK;
				}
				// else if stopped reading row then there is another row
				else if (!rowOpen)
				{
					lastLayer->r++;
					lastLayer->c = 0;
					lastLayer->neurons = (float **)realloc(lastLayer->neurons, lastLayer->r * sizeof(float *));
					lastLayer->neurons[lastLayer->r - 1] = (float *)malloc(sizeof(float));
				}

				i++;
			}
		BREAK:
			break;
		}
	}

	return l;
}

int main(int argc, char **argv)
{
	float *inputs;	// inputs
	Layer *hn, on;	// hidden layers, outputs respectievly
	int lcnt = 0;	// hidden layers count
	int inCnt = 0;	// count of inputs

	// open file
	printf("Enter name of file with weights: ");
	// fgets(fn, MAXFNAME, stdin);
	// fn[strlen(fn) - 1] = 0;
	printf("Trying to open \"%s\"\n", fn);
	if (!(rfd = fopen(fn, "r")))
	{
		fprintf(stderr, "Failed to open \"%s\"\n", fn);
		return 1;
	}

	hn = NULL;
	hn = readWeights(hn, &lcnt, &inCnt);

	printf("Neural network successfully copied\n\n");
	fflush(stdout);
	fclose(rfd);

	// count of inputs is number of rown in first layer
	inCnt = hn[0].r;
	inputs = (float *)malloc(inCnt * sizeof(float));

	// allocating memory for w_sums
	for (size_t i = 0; i < lcnt; i++)
		hn[i].wsums = (float *)malloc(hn[i].c * sizeof(float));

	// last layer is output layer
	on = hn[lcnt - 1];
	lcnt--;

	// give a work to NN
	while (1)
	{
		printf("Please, write a filename of inputs or \"exit\": ");
		fgets(inputsName, MAXFNAME, stdin);
		inputsName[strlen(inputsName) - 1] = 0;

		if (strcmp(inputsName, "exit") == 0)
			break;

		if (!(rfd = fopen(inputsName, "r")))
		{
			fprintf(stderr, "Failed to open \"%s\"\n\n", inputsName);
			continue;
		}

		// reading inputs
		while(readInputs(inputs, inCnt))
		{
			// if have hidden layers
			if (lcnt > 0)
			{
				// calculating w_sum for each result in 1st hid layer
				dot_prod(inputs, hn);

				// calculating w_sum for each result in other hid layers
				for (int i = 1; i < lcnt; i++)
					dot_prod(hn[i - 1].wsums, hn + i);

				// calculating w_sum for each result in output layer
				dot_prod(hn[lcnt - 1].wsums, &on);
			}
			else
			{
				dot_prod(inputs, &on);
			}

			// printig outputs
			for (int i = 0; i < on.c; i++)
				printf("output[%d] = %f\n", i, on.wsums[i]);
			printf("\n");
		}
		
		fclose(rfd);
	}

	// free all malloced memory
	free(inputs);	// input neurons

	for (int i = 0; i < lcnt; i++)
	{
		for (int j = 0; j < hn[i].r; j++)
			free(hn[i].neurons[j]);	// weights in columns
		free(hn[i].neurons);		// columns
		free(hn[i].wsums);			// results
	}
	free(hn);	// array of hidden layers

	for (int i = 0; i < on.r; i++)
		free(on.neurons[i]);	// output weights in columns
	free(on.neurons);			// output columns
	free(on.wsums);				// output results

	return 0;
}
