#include <iostream>
#include <vector>
#include <map>
#include <cmath>

using namespace std;

// ������
double entropy(const map<int, double>& probabilities) {
    double h = 0.0;
    for (const auto& [event, prob] : probabilities) {
        if (prob > 0) { // ���Ը���Ϊ����¼�
            h -= prob * log2(prob);
        }
    }
    return h;
}

// ���������� H(X|Y)
double conditional_entropy(const map<int, double>& joint_probabilities, const map<int, double>& probabilities_Y) {
    double h = 0.0;
    for (const auto& [event_y, prob_y] : probabilities_Y) {
        if (prob_y > 0) {
            for (const auto& [event_x, joint_prob] : joint_probabilities) {
                if (joint_prob > 0) {
                    double conditional_prob = joint_prob / prob_y; // P(X|Y)
                    h -= conditional_prob * log2(conditional_prob);
                }
            }
        }
    }
    return h;
}

// ���㻥��Ϣ I(X; Y)
double mutual_information(const map<int, double>& probabilities_X,
                          const map<int, double>& probabilities_Y,
                          const map<pair<int, int>, double>& joint_probabilities) {
    double I = 0.0;

    // ���� I(X; Y) = H(X) + H(Y) - H(X, Y)
    double H_X = entropy(probabilities_X);
    double H_Y = entropy(probabilities_Y);
    
    double H_XY = 0.0;
    for (const auto& [event, joint_prob] : joint_probabilities) {
        if (joint_prob > 0) {
            H_XY -= joint_prob * log2(joint_prob);
        }
    }

    I = H_X + H_Y - H_XY;
    return I;
}

int main() {
    map<int, double> probabilities_X = {{0, 0.5}, {1, 0.5}};
    map<int, double> probabilities_Y = {{0, 0.6}, {1, 0.4}};
    
    // ���ϸ��ʷֲ� (P(X,Y))
    map<pair<int, int>, double> joint_probabilities = {
        {{0, 0}, 0.3},   // P(X=0, Y=0)
        {{0, 1}, 0.2},   // P(X=0, Y=1)
        {{1, 0}, 0.3},   // P(X=1, Y=0)
        {{1, 1}, 0.2}    // P(X=1, Y=1)
    };

    // ������
    double H_X = entropy(probabilities_X);
    double H_Y = entropy(probabilities_Y);
    
    // ���������� H(X|Y)
    double H_X_given_Y = conditional_entropy(joint_probabilities, probabilities_Y);
    
    // ���㻥��Ϣ I(X; Y)
    double I_XY = mutual_information(probabilities_X, probabilities_Y, joint_probabilities);

    cout << "H(X) = " << H_X << endl;
    cout << "H(Y) = " << H_Y << endl;
    cout << "H(X|Y) = " << H_X_given_Y << endl;
    cout << "I(X; Y) = " << I_XY << endl;

    return 0;
}