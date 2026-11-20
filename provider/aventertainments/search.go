package aventertainments

import (
	"strconv"
	"strings"

	"github.com/metatube-community/metatube-sdk-go/model"
)

// searchMovieFromIndex answers a search through the offline table.
//
// Any doubt returns false so the caller falls back to the site's own search: a
// table miss, a failed detail page, invalid metadata and a number mismatch all
// take that path. The detail page is authoritative — it is what the table was
// built from, and the number it prints is what must match.
func (ave *AVE) searchMovieFromIndex(keyword string) (*model.MovieSearchResult, bool) {
	pro, ok := lookupPro(keyword)
	if !ok {
		return nil, false
	}
	info, err := ave.GetMovieInfoByID(strconv.Itoa(pro))
	if err != nil || info == nil || !info.IsValid() {
		return nil, false
	}
	if !sameNumber(info.Number, keyword) {
		return nil, false
	}
	return info.ToSearchResult(), true
}

// sameNumber compares two numbers the way the site does: surrounding space and
// letter case do not distinguish products, separators do.
func sameNumber(a, b string) bool {
	return strings.EqualFold(strings.TrimSpace(a), strings.TrimSpace(b))
}
