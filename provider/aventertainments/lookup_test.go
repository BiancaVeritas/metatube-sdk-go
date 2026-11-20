package aventertainments

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestLookupProIsExactForNumericKeys pins the one shape that bites: a handful of
// table keys are bare digits because the crawl read the number off a thumbnail,
// while the detail page prints `DVD007`. The lookup must answer with that key,
// and the caller's number check is what rejects the mismatch.
func TestLookupProIsExactForNumericKeys(t *testing.T) {
	pro, ok := lookupPro("07")
	assert.True(t, ok)
	assert.Positive(t, pro)
	// `DVD007` is what the detail page prints; it is not a table key.
	_, ok = lookupPro("DVD007")
	assert.False(t, ok)

	assert.False(t, sameNumber("DVD007", "07"))
}
