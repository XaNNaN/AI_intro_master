#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <png.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define PNG_BYTES_TO_CHECK 4
#define Nh 4
#define Nv 4
#define N Nh * Nv
#define L 16
#define M N
#define T 100000
#define W_START_WIDTH 0.5
#define ETTA 0.01
#define EPOCH_COUNT 2000
#define PROGRESS_BAR_EL_COUNT 50
#define MAX_ERROR 0
#define MAX_PIX_ERROR 2
#define MAX_ERORR_DELTA 1e-8

png_structp png_read_ptr;
png_infop info_read_ptr;
png_structp png_write_ptr;
png_infop info_write_ptr;
png_uint_32 width, height;
png_bytep * row_pointers;
png_bytep * original_image;
png_bytep * currnet_res;
int bit_depth, color_type, interlace_type;
int compression_type, filter_method;
int row, col;
char sigBuf[PNG_BYTES_TO_CHECK];

FILE * fpIn;
FILE * fpOut;

void readPng(char *, int );
void writePng(char * );

float ** W_1;
float ** W_2;
float ** W_1_new;
float ** W_2_new;
unsigned char input[N];
int Gval[L];
int Y[M];
int Gval_new[L];
int Y_new[M];

float etta = ETTA;

double calculate_error(){
    double error = 0;
    for (int row = 0; row < height; row++ ){
        for (int col = 0; col < width; col++){
            png_byte color = currnet_res[row][col];
            int int_col = (int) currnet_res[row][col];
            bool check = color == int_col;
            double diff = (double) currnet_res[row][col] - original_image[row][col];
            error = error + pow(diff, 2);
        }
    }
    return error;
}

void trainOnSegmet(int h_seg, int v_seg) {
    int t = 0;
    int goodTryCount = 0;
    int delatMoreThanEps = 1;

    // copy data from png to X vector
    for (int h = 0; h < Nh; h++)
        for (int v = 0; v < Nv; v++)
            input[Nv * h + v] = row_pointers[Nv * v_seg + v][Nh * h_seg + h];

    while (t < T && delatMoreThanEps) { // add delta < e
        t++;
        // cal Y and G
        for (int l = 0; l < L; l++) {
            int g_l = 0;
            for (int j = 0; j < N; j++) {
                g_l += W_1[l][j] * input[j];
            }
            Gval[l] = g_l;
        }
        for (int i = 0; i < M; i++) {
            int y_i = 0;
            for (int l = 0; l < L; l++) {
                y_i += W_2[i][l] * Gval[l];
            }
            Y[i] = y_i;
        }
        // end calc Y and G
        // dW_2(il)
        for (int i = 0; i < M; i++) {
            for (int l = 0; l < L; l++) {
                float grad = (Y[i] - input[i]) * Gval[l];
                W_2_new[i][l] = W_2[i][l] - etta * grad;
            }
        }
        // dW_1(lj)
        for (int l = 0; l < L; l++) {
            for (int j = 0; j < N; j++) {
                float grad = 0;
                for (int i = 0; i < M; i++) {
                    grad += (Y[i] - input[i]) * W_2[i][l] * input[j];
                }
                W_1_new[l][j] = W_1[l][j] - etta * grad;
            }
        }
        // calc new Y and G
        for (int l = 0; l < L; l++) {
            int g_l = 0;
            for (int j = 0; j < N; j++) {
                g_l += W_1_new[l][j] * input[j];
            }
            Gval_new[l] = g_l;
        }
        for (int i = 0; i < M; i++) {
            int y_i = 0;
            for (int l = 0; l < L; l++) {
                y_i += W_2_new[i][l] * Gval_new[l];
            }
            Y_new[i] = y_i;
        }
        // end clac new Y and G
        // calc oldE
        float oldE = 0;
        for (int i = 0; i < N; i++) {
            float delta = Y[i] - input[i];
            oldE += delta * delta;
        }
        // oldE /= 2;
        // end calc oldE
        // calc newE
        float newE = 0;
        for (int i = 0; i < N; i++) {
            float delta = Y_new[i] - input[i];
            newE += delta * delta;
        }
        // newE /= 2;
        // end calc newE
        float deltaE = oldE - newE;
        if (deltaE < 0) {
            etta /= 2.; // TODO constant
            goodTryCount = 0;
        } else {
            // etta = etta * 2;
            goodTryCount++;
            // swap
            float ** tmpPtr_1 = W_1;
            float ** tmpPtr_2 = W_2;
            W_1 = W_1_new;
            W_2 = W_2_new;
            W_1_new = tmpPtr_1;
            W_2_new = tmpPtr_2;
            // end swap
            if (deltaE < MAX_ERORR_DELTA) // TODO constant
                // printf("%lf\n", deltaE);
                delatMoreThanEps = 0;
            if (goodTryCount == 2) { // TODO constant
                etta *= 2.;
                goodTryCount = 0;
            }
        }
    }
    return;
}

void useNNetOnSegmet(int h_seg, int v_seg, int mode) {
    // copy data from png to X vector
    for (int h = 0; h < Nh; h++) {
        for (int v = 0; v < Nv; v++) {
            input[Nv * h + v] = row_pointers[Nv * v_seg + v][Nh * h_seg + h];
        }
    }

    for (int l = 0; l < L; l++) {
        int g_l = 0;
        for (int j = 0; j < N; j++) {
            g_l += W_1[l][j] * input[j];
        }

        Gval[l] = g_l;
    }

    for (int i = 0; i < M; i++) {
        int y_i = 0;
        for (int l = 0; l < L; l++) {
            y_i += W_2[i][l] * Gval[l];
        }
        Y[i] = y_i;
    }

    for (int h = 0; h < Nh; h++) {
        for (int v = 0; v < Nv; v++) {
            if (mode == 0)
                row_pointers[Nv * v_seg + v][Nh * h_seg + h] = Y[Nv * h + v];   
            else 
                currnet_res[Nv * v_seg + v][Nh * h_seg + h] = Y[Nv * h + v];
        }

    }

    return;
}

void setInitialW(float ** W, int I, int J) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            W[i][j] = (float) rand() / (float)(RAND_MAX / W_START_WIDTH) -
                (W_START_WIDTH / 2.0);
        }
    }
}

void printW(float ** W, int I, int J) {
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) 
            printf("%.2lf ", W[i][j]);
        printf("\n");
    }
}

void apply_network(int h_segCount, int v_segCount){
    for (int h_seg = 0; h_seg < h_segCount; h_seg++) {
        for (int v_seg = 0; v_seg < v_segCount; v_seg++) {
            useNNetOnSegmet(h_seg, v_seg, 1);
        }
    }
}

int main(int argc, char * argv[]) {
    double min_error;
    double med_error_per_pixel;
    double min_med_error;

    srand(time(NULL));

    if (argc < 3) {
        printf("where is file names?\n");
        exit(0);
    }

    readPng(argv[1], 0);
    readPng(argv[1], 1);
    readPng(argv[1], 2);

    if ((W_1 = (float ** ) calloc(L, sizeof(float * ))) == NULL) {
        printf("Не хватает памяти W_1*.");
        exit(1);
    }

    for (int i = 0; i < L; i++) {
        if ((W_1[i] = (float * ) calloc(N, sizeof(float))) == NULL) {
            printf("Не хватает памяти W_1.");
            exit(1);
        }
    }

    if ((W_2 = (float ** ) calloc(M, sizeof(float * ))) == NULL) {
        printf("Не хватает памяти W_2*.");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        if ((W_2[i] = (float * ) calloc(L, sizeof(float))) == NULL) {
            printf("Не хватает памяти W_2.");
            exit(1);
        }

    }

    if ((W_1_new = (float ** ) calloc(L, sizeof(float * ))) == NULL) {
        printf("Не хватает памяти W_1*.");
        exit(1);
    }

    for (int i = 0; i < L; i++) {
        if ((W_1_new[i] = (float * ) calloc(N, sizeof(float))) == NULL) {
            printf("Не хватает памяти W_1.");
            exit(1);
        }

    }

    if ((W_2_new = (float ** ) calloc(M, sizeof(float * ))) == NULL) {
        printf("Не хватает памяти W_2*.");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        if ((W_2_new[i] = (float * ) calloc(L, sizeof(float))) == NULL) {
            printf("Не хватает памяти W_2.");
            exit(1);
        }
    }

    setInitialW(W_1, L, N);
    setInitialW(W_2, M, L);
    if (height % Nv) {
        printf("Вертикальный размер не кратен Nv.\n");
        exit(0);
    }
    if (width % Nh) {
        printf("Горизонтальный размер не кратен Nh.\n");
        exit(0);
    }
    const int h_segCount = width / Nh;
    const int v_segCount = height / Nv;
    printf("\n\n");
    // network training
    apply_network(h_segCount, v_segCount);
    double E = calculate_error();
    min_error = E;
    med_error_per_pixel = sqrt(E / width / height);
    min_med_error = med_error_per_pixel;
    int epoch = 1;
    while( (E > MAX_ERROR) && (epoch < EPOCH_COUNT) && (med_error_per_pixel > MAX_PIX_ERROR)){
        epoch ++;
    // for (int epoch = 1; epoch <= EPOCH_COUNT; epoch++) {
        for (int h_seg = 0; h_seg < h_segCount; h_seg++) {
            for (int v_seg = 0; v_seg < v_segCount; v_seg++) {
                trainOnSegmet(h_seg, v_seg);
            }
        }
        // find new Error
        apply_network(h_segCount, v_segCount);
        E = calculate_error();
        med_error_per_pixel = sqrt(E / width / height);
        if (min_error > E)
            min_error = E;
        if ( min_med_error > med_error_per_pixel)
            min_med_error = med_error_per_pixel;

        // print progress bar
        // TODO: delete progress bar
        int progress = 100 * epoch / EPOCH_COUNT;
        int pLine = progress * PROGRESS_BAR_EL_COUNT / 100;
        int pc = 0;
        char esc[4];
        esc[0] = 27;
        esc[1] = '[';
        esc[2] = 'A'; // esc[3] = 'K';
        write(1, esc, 3);
        while (pc < PROGRESS_BAR_EL_COUNT) {
            if (pc < pLine)
                printf("#");
            else
                printf("_");
            pc++;
        }
        printf("| %d%%\n", progress);
    }
    printf("W_1\n");
    printW(W_1, L, M);
    printf("\nW_2\n");
    printW(W_2, M, L);

    // decompressing
    for (int h_seg = 0; h_seg < h_segCount; h_seg++) {
        for (int v_seg = 0; v_seg < v_segCount; v_seg++) {
            useNNetOnSegmet(h_seg, v_seg, 0);
        }
    }

    for (int i = 0; i < L; i++) 
        free(W_1[i]);
    free(W_1);

    for (int i = 0; i < M; i++)
        free(W_2[i]);
    free(W_2);

    for (int i = 0; i < L; i++) 
        free(W_1_new[i]);
    free(W_1_new);

    for (int i = 0; i < M; i++) 
        free(W_2_new[i]);
    free(W_2_new);

    writePng(argv[2]);
    printf("Min error: %lf\n", min_error);
    printf("Min medium error per pixel: %lf\n", min_med_error);
    return 0;
}

void readPng(char * fileName, int mode) {

    fpIn = stdin;

    if ((fpIn = fopen(fileName, "r")) == NULL) {
        perror(fileName);
        exit(1);
    };

    if (fread(sigBuf, 1, PNG_BYTES_TO_CHECK, fpIn) != PNG_BYTES_TO_CHECK) {
        fclose(fpIn);
        exit(2);
    };

    /* Проверка первых PNG_BYTES_TO_CHECK байт заголовка png-файла */

    if (png_sig_cmp(sigBuf, (png_size_t) 0, PNG_BYTES_TO_CHECK)) {
        fclose(fpIn);
        exit(3);
    };

    png_read_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_read_ptr == NULL) {
        fclose(fpIn);
        exit(2);
    };

    info_read_ptr = png_create_info_struct(png_read_ptr);

    if (info_read_ptr == NULL) {
        fclose(fpIn);
        png_destroy_read_struct( & png_read_ptr, NULL, NULL);
        exit(3);
    };

    if (setjmp(png_jmpbuf(png_read_ptr))) {
        png_destroy_read_struct( & png_read_ptr, & info_read_ptr, NULL);
        fclose(fpIn);
        exit(4);
    };

    png_init_io(png_read_ptr, fpIn);

    /* Информировать о том, что PNG_BYTES_TO_CHECK байт уже прочитано
     */

    png_set_sig_bytes(png_read_ptr, PNG_BYTES_TO_CHECK);
    png_read_png(png_read_ptr, info_read_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    fclose(fpIn);

    png_get_IHDR(png_read_ptr, info_read_ptr, & width, & height, & bit_depth, & color_type, & interlace_type, & compression_type, & filter_method);
    printf("Ширина = %d, высота = %d\n", width, height);
    printf("Тип цвета = %d, глубина цвета = %d\n", color_type, bit_depth);
    printf("Количество байт в строке = %ld\n", png_get_rowbytes(png_read_ptr, info_read_ptr));

    if (color_type != 0) {
        fprintf(stderr, "Не умею работать с типом цвета %d (только 0)\n ", color_type);
        exit(11);
    };

    if (bit_depth != 8) {
        fprintf(stderr, "Не умею работать с глубиной цвета %d (только 8)\n ", bit_depth);
        exit(11);
    };

    if (mode == 0)
        row_pointers = png_get_rows(png_read_ptr, info_read_ptr);
    else if (mode == 1)
        original_image = png_get_rows(png_read_ptr, info_read_ptr);
    else 
        currnet_res = png_get_rows(png_read_ptr, info_read_ptr);

    for (col = 0; col < width; col++) {
        /* if (row_pointers[row][col] < 85) {
            printf(" ");
        } else if (row_pointers[row][col] < 170) {
            printf("-");
        } else {
            printf("+");
        } */
        row_pointers[row][col] ^= 0xFF;
    }
}

void writePng(char * fileName) {

    fpOut = stdout;

    if ((fpOut = fopen(fileName, "w")) == NULL) {
        perror(fileName);
        exit(5);
    };

    png_write_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_write_ptr == NULL) {
        fclose(fpOut);
        exit(6);
    };

    info_write_ptr = png_create_info_struct(png_write_ptr);

    if (info_write_ptr == NULL) {
        fclose(fpOut);
        png_destroy_write_struct( & png_write_ptr, NULL);
        exit(7);
    };

    if (setjmp(png_jmpbuf(png_write_ptr))) {
        fclose(fpOut);
        png_destroy_write_struct( & png_write_ptr, & info_write_ptr);
        exit(8);
    };

    png_set_IHDR(png_write_ptr, info_write_ptr, width, height, bit_depth, color_type, interlace_type, compression_type, filter_method);
    png_set_rows(png_write_ptr, info_write_ptr, row_pointers);
    png_init_io(png_write_ptr, fpOut);
    png_write_png(png_write_ptr, info_write_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    fclose(fpOut);

    return;
}