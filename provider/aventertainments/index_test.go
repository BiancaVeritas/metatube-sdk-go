package aventertainments

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestProIndexShape guards the embedded table: it is generated out of tree, so a
// truncated or badly keyed file would silently cost every search.
func TestProIndexShape(t *testing.T) {
	index := proIndex()
	assert.NotEmpty(t, index)
	for number, pro := range index {
		assert.NotEmpty(t, number)
		assert.Positive(t, pro)
	}
}

func TestLookupPro(t *testing.T) {
	for _, unit := range []struct {
		in   string
		want int
		ok   bool
	}{
		// The two shapes the site uses for the same product family.
		{"SKYHD-059", 45847, true},
		{"skyhd-059", 45847, true},
		{"  SKYHD-059  ", 45847, true},
		{"ASM-892034", 162498, true},
		{"DR009", 80, true},
		{"NOSUCH-000", 0, false},
		{"", 0, false},
		{"   ", 0, false},
	} {
		got, ok := lookupPro(unit.in)
		assert.Equal(t, unit.ok, ok, "input %q", unit.in)
		if unit.ok {
			assert.Equal(t, unit.want, got, "input %q", unit.in)
		}
	}
}

// TestLookupProKeepsSeparators documents why the lookup is not normalized: the
// site treats the two spellings as different products, and both are in the
// table, so folding them would answer with the wrong pro.
func TestLookupProKeepsSeparators(t *testing.T) {
	withDash, okDash := lookupPro("AND-001")
	without, okPlain := lookupPro("AND001")
	if okDash && okPlain {
		assert.NotEqual(t, withDash, without)
	}
}

func TestSameNumber(t *testing.T) {
	for _, unit := range []struct {
		a, b string
		want bool
	}{
		{"SKYHD-059", "skyhd-059", true},
		{"SKYHD-059", " SKYHD-059 ", true},
		{"AND-001", "AND001", false},
		{"F-B001", "FB001", false},
		{"SKYHD-059", "SKYHD-060", false},
	} {
		assert.Equal(t, unit.want, sameNumber(unit.a, unit.b), "%q vs %q", unit.a, unit.b)
	}
}
