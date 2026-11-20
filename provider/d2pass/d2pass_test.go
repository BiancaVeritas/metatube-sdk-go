package d2pass

import (
	_ "embed"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/metatube-community/metatube-sdk-go/model"
)

// payload is a slice of the Next.js streamed payload of one product page,
// captured verbatim from the site.
//
//go:embed testdata/product_payload.txt
var payload string

func TestFillFromPayload(t *testing.T) {
	require.NotEmpty(t, payload)
	info := &model.MovieInfo{ID: "50162"}
	fillFromPayload(info, payload)

	assert.Equal(t, "50162", info.ID)
	assert.Equal(t, "HEYZO-0333", info.Number)
	assert.Equal(t, "神尻妄想ナース", info.Title)
	assert.Equal(t, "HEYZO", info.Maker)
	assert.Equal(t, []string{"一ノ瀬ルカ"}, []string(info.Actors))
	assert.NotEmpty(t, info.Summary)
	assert.NotEmpty(t, info.Genres)
	assert.NotEmpty(t, info.CoverURL)
	assert.Contains(t, info.CoverURL, "50162.webp")
	assert.Positive(t, info.Runtime)
	assert.False(t, time.Time(info.ReleaseDate).IsZero())
}

func TestFillFromPayloadMissingProduct(t *testing.T) {
	info := &model.MovieInfo{ID: "1"}
	fillFromPayload(info, "<html><body>no product here</body></html>")
	// Nothing is invented: a page without the payload leaves the info as it was.
	assert.Equal(t, "1", info.ID)
	assert.Empty(t, info.Title)
	assert.Empty(t, info.Number)
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
		{"HEYZO-3381", "3381", true},
		{"HEYZO-0000", "", false},
		{"HEYZO-", "", false},
		{"HEYZO-abc", "", false},
		{"ABP-333", "", false},
		{"", "", false},
		{"  HEYZO-0042  ", "0042", true},
	} {
		got, ok := heyzoDigits(unit.in)
		assert.Equal(t, unit.ok, ok, "input %q", unit.in)
		assert.Equal(t, unit.want, got, "input %q", unit.in)
	}
}

func TestProductID(t *testing.T) {
	for _, unit := range []struct {
		in, want string
		ok       bool
	}{
		{"HEYZO-0333", "50162", true},
		{"50162", "50162", true},
		{"HEYZO-9999", "", false},
		{"abc", "", false},
		{"", "", false},
	} {
		got, ok := productID(unit.in)
		assert.Equal(t, unit.ok, ok, "input %q", unit.in)
		if unit.ok {
			assert.Equal(t, unit.want, got, "input %q", unit.in)
		}
	}
}

func TestNormalizeMovieKeyword(t *testing.T) {
	d2 := New()
	for _, unit := range []struct{ in, want string }{
		{"HEYZO-0333", "HEYZO-0333"},
		{"heyzo_333", "HEYZO_333"},
		{"ABP-030", ""},
		{"50162", ""},
	} {
		assert.Equal(t, unit.want, d2.NormalizeMovieKeyword(unit.in), "input %q", unit.in)
	}
}
