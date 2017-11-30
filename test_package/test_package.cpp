#include <tiny_dnn/tiny_dnn.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_net(network<sequential>& nn)
{
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    core::backend_t backend_type = core::backend_t::tiny_dnn;

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6,  // C1, 1@32x32-in, 6@28x28-out
            padding::valid, true, 1, 1, backend_type)
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16, // C3, 6@14x14-in, 16@10x10-in
            connection_table(tbl, 6, 16),
            padding::valid, true, 1, 1, backend_type)
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120, // C5, 16@5x5-in, 120@1x1-out
            padding::valid, true, 1, 1, backend_type)
       << fully_connected_layer<tan_h>(120, 10,        // F6, 120-in, 10-out
            true, backend_type)
    ;
}

int main(void)
{
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);

    return 0;
}
