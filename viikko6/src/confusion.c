#include <zephyr/kernel.h>
#include <math.h>
#include "confusion.h"
#include "adc.h"

int CP[6][3] = {
    {1320.444444, 1630.296296, 1629.148148},
    {1969.857143, 1602.607143, 1620.428571},
    {1623.035714, 1282.500000, 1609.678571},
    {1664.851852, 1948.481481, 1642.592593},
    {1640.785714, 1633.642857, 1312.321429},
    {1644.892857, 1620.178571, 1956.250000}
};

int measurements[6][3] = {
    {1320.444444, 1630.296296, 1629.148148},
    {1969.857143, 1602.607143, 1620.428571},
    {1623.035714, 1282.500000, 1609.678571},
    {1664.851852, 1948.481481, 1642.592593},
    {1640.785714, 1633.642857, 1312.321429},
    {1644.892857, 1620.178571, 1956.250000}
};

int CM[6][6] = {0};

void printConfusionMatrix(void) {
    printk("Confusion matrix = \n");
    printk("   cp1 cp2 cp3 cp4 cp5 cp6\n");
    for (int i = 0; i < 6; i++) {
        printk("cp%d %d   %d   %d   %d   %d   %d\n", i+1, CM[i][0], CM[i][1], CM[i][2], CM[i][3], CM[i][4], CM[i][5]);
    }
}

int calculateDistance(int x1, int y1, int z1, int x2, int y2, int z2) {
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2)); // sama ku python mutta ceesä
}

int calculateDistanceToAllCentrePointsAndSelectWinner(int x, int y, int z) {
    int minDistance = INT_MAX;
    int winner = -1;
    for (int i = 0; i < 6; i++) {
        int dist = calculateDistance(x, y, z, CP[i][0], CP[i][1], CP[i][2]);
        if (dist < minDistance) {
            minDistance = dist;
            winner = i;
        }
    }
    return winner;
}

void makeOneClassificationAndUpdateConfusionMatrix(int direction) {


     for (int i = 0; i < 100; i++) {
    struct Measurement m = readADCValue(); // mittqaukset
    int x = m.x;
    int y = m.y;
    int z = m.z;
    

    int winner = calculateDistanceToAllCentrePointsAndSelectWinner(x, y, z);
    if (winner >= 0) {
        CM[direction][winner]++;
    }
     }
}


void makeHundredFakeClassifications(void) {
    for (int i = 0; i < 100; i++) {
        int randomIndex = rand() % 6;
        makeOneClassificationAndUpdateConfusionMatrix(randomIndex);
    }
}

void resetConfusionMatrix(void) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            CM[i][j] = 0;
        }
    }
}
