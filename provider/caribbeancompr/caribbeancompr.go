package caribbeancompr

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/metatube-community/metatube-sdk-go/model"
	"github.com/metatube-community/metatube-sdk-go/provider"
	"github.com/metatube-community/metatube-sdk-go/provider/caribbeancom/core"
)

var (
	_ provider.MovieProvider = (*CaribbeancomPremium)(nil)
	_ provider.MovieReviewer = (*CaribbeancomPremium)(nil)
)

const (
	Name     = "CaribbeancomPR"
	Priority = 1000 - 1
)

const (
	baseURL  = "https://www.caribbeancompr.com/"
	movieURL = "https://www.caribbeancompr.com/moviepages/%s/index.html"
)

// pageID is the site's own movie id, `YYMMDD_NNN`.
var pageID = regexp.MustCompile(`^\d{6}_\d{3}$`)

type CaribbeancomPremium struct {
	*core.Core
}

func New() *CaribbeancomPremium {
	return &CaribbeancomPremium{
		Core: (&core.Core{
			BaseURL:         baseURL,
			MovieURL:        movieURL,
			DefaultName:     Name,
			DefaultPriority: Priority,
			DefaultMaker:    "カリビアンコムプレミアム",
		}).Init(),
	}
}

// NormalizeMovieID accepts the site's page ids and HEYZO numbers.
//
// The site lists HEYZO works under page ids and never prints the HEYZO number,
// so a HEYZO request is served through the offline table, which is the only
// thing tying the two together. Anything else is not this provider's business.
func (carib *CaribbeancomPremium) NormalizeMovieID(id string) string {
	id = strings.TrimSpace(id)
	if pageID.MatchString(id) {
		return id
	}
	if digits, ok := heyzoDigits(id); ok {
		return "HEYZO-" + digits
	}
	return ""
}

// GetMovieInfoByID resolves a HEYZO number through the table and then reads the
// page, so the numbers the index knows still get full metadata from the site.
func (carib *CaribbeancomPremium) GetMovieInfoByID(id string) (*model.MovieInfo, error) {
	if page, ok := heyzoPageID(id); ok {
		return carib.GetMovieInfoByURL(fmt.Sprintf(movieURL, page))
	}
	return carib.Core.GetMovieInfoByID(id)
}

// GetMovieInfoByURL reports the HEYZO number for works the table knows instead
// of leaving the site's page id in the number field, and applies the verified
// title where the site's own title is wrong.
func (carib *CaribbeancomPremium) GetMovieInfoByURL(rawURL string) (*model.MovieInfo, error) {
	info, err := carib.Core.GetMovieInfoByURL(rawURL)
	if err != nil || info == nil {
		return info, err
	}
	number, overrideTitle, ok := heyzoNumberForPage(info.ID)
	if !ok {
		return info, nil
	}
	info.Number = number
	if overrideTitle != "" {
		info.Title = overrideTitle
	}
	return info, nil
}

func init() {
	provider.Register(Name, New)
}
