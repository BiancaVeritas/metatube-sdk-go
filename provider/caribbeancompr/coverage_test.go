package caribbeancompr

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestHeyzoTablesCoverDifferentRanges documents why both HEYZO tables exist:
// each source knows works the other does not, so neither replaces the other.
func TestHeyzoTablesCoverDifferentRanges(t *testing.T) {
	table, _ := heyzoLoad()
	assert.NotEmpty(t, table)
	// Spot checks: a work only CaribbeancomPR carries, and the reverse case is
	// covered by the D2PASS package's own table.
	assert.Contains(t, table, "1426")
	assert.NotContains(t, table, "0333")
}
