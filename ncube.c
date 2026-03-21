#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <inttypes.h>

// prints the n-dim unit cube.

int main(int argc, char **argv) {
    // accept a single argument, the dimension
    assert(argc == 2);
    int n = atoi(argv[1]);
    assert((1 <= n) && (n <= 10));

    // vertices corresponds to bits
    uint64_t upper = (n == 64) ? UINT64_MAX : ((uint64_t)1 << n) - 1;

    for (uint64_t i = 0; i<=upper; ++i) {
        printf("[");
        for (int j=n-1; j>=0; --j)
            printf("%" PRIu64 ",", 1&(i>>j));
        printf("1], "); // end with 1 (to homogenize)
    }
    printf("\n");

    return 0;
}
