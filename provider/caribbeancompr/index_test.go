package caribbeancompr

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestHeyzoIndexShape guards the embedded table: it is generated out of tree, so
// a truncated or badly keyed file would silently cost every HEYZO lookup.
func TestHeyzoIndexShape(t *testing.T) {
	table, ids := heyzoLoad()
	assert.NotEmpty(t, table)
	assert.Len(t, table, len(ids), "every row must map back to a page id")
	for digits, entry := range table {
		assert.Len(t, digits, 4, "keys are zero-padded to four digits")
		assert.NotEmpty(t, entry.ID)
		assert.Equal(t, digits, ids[entry.ID])
	}
}

func TestHeyzoDigits(t *testing.T) {
	for _, unit := range []struct {
		in, want string
		ok       bool
	}{
		{"HEYZO-0333", "0333", true},
		{"heyzo-333", "0333", true},
		{"HEYZO_333", "0333", true},
		{"HEYZO333", "0333", true},
		{"HEYZO-1426", "1426", true},
		{"HEYZO-0000", "", false},
		{"HEYZO-", "", false},
		{"HEYZO-abc", "", false},
		{"ABP-030", "", false},
		{"", "", false},
	} {
		got, ok := heyzoDigits(unit.in)
		assert.Equal(t, unit.ok, ok, "input %q", unit.in)
		assert.Equal(t, unit.want, got, "input %q", unit.in)
	}
}

func TestHeyzoPageID(t *testing.T) {
	for _, unit := range []struct {
		in, want string
		ok       bool
	}{
		{"HEYZO-0333", "", false}, // not in the CaribbeancomPR table
		{"HEYZO-1426", "072917_001", true},
		{"HEYZO-0004", "120120_003", true},
		{"HEYZO-9999", "", false},
	} {
		got, ok := heyzoPageID(unit.in)
		assert.Equal(t, unit.ok, ok, "input %q", unit.in)
		assert.Equal(t, unit.want, got, "input %q", unit.in)
	}
}

// TestHeyzoNumberForPage pins the reverse lookup, including the verified title.
func TestHeyzoNumberForPage(t *testing.T) {
	for _, unit := range []struct {
		pageID, number, override string
		ok                       bool
	}{
		{"120120_003", "HEYZO-0004", "", true},
		{"999999_999", "", "", false},
	} {
		number, override, ok := heyzoNumberForPage(unit.pageID)
		assert.Equal(t, unit.ok, ok, "page %q", unit.pageID)
		if unit.ok {
			assert.Equal(t, unit.number, number, "page %q", unit.pageID)
			assert.Equal(t, unit.override, override, "page %q", unit.pageID)
		}
	}
}

// TestHeyzoOverrideTitle pins the case the table exists for: the site labels a
// sequel volume with the wrong number in its title.
func TestHeyzoOverrideTitle(t *testing.T) {
	_, override, ok := heyzoNumberForPage("072917_001")
	assert.True(t, ok)
	assert.Contains(t, override, "vol.38")
}

func TestNormalizeMovieID(t *testing.T) {
	carib := New()
	for _, unit := range []struct{ in, want string }{
		{"052121_002", "052121_002"},
		{"HEYZO-1426", "HEYZO-1426"},
		{"heyzo_1426", "HEYZO-1426"},
		{"HEYZO-0333", "HEYZO-0333"}, // resolves through the table elsewhere
		{"ABP-030", ""},
		{"", ""},
	} {
		assert.Equal(t, unit.want, carib.NormalizeMovieID(unit.in), "input %q", unit.in)
	}
}
