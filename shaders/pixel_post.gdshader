shader_type canvas_item;
// Pixel-art post-process: snap the frame to a chunky pixel grid, quantize the
// palette, and (optionally) apply ordered Bayer dithering between the bands —
// the classic retro look, applied over the whole 3D render.

uniform sampler2D screen_tex : hint_screen_texture, filter_nearest;
uniform int pixel_size : hint_range(1, 12) = 4;   // screen pixels per art pixel
uniform int levels : hint_range(2, 16) = 6;       // colour steps per channel
uniform bool dither = true;

// 4x4 Bayer matrix, normalized to [0, 1)
const mat4 BAYER = mat4(
	vec4( 0.0,  8.0,  2.0, 10.0),
	vec4(12.0,  4.0, 14.0,  6.0),
	vec4( 3.0, 11.0,  1.0,  9.0),
	vec4(15.0,  7.0, 13.0,  5.0)) / 16.0;

void fragment() {
	vec2 sz = vec2(textureSize(screen_tex, 0));
	// centre of the chunky pixel this fragment belongs to
	vec2 cell = floor(SCREEN_UV * sz / float(pixel_size));
	vec2 uv = (cell * float(pixel_size) + float(pixel_size) * 0.5) / sz;
	vec3 c = texture(screen_tex, uv).rgb;
	if (dither) {
		// threshold varies per chunky pixel, pushing colours across band edges
		float t = BAYER[int(cell.x) % 4][int(cell.y) % 4];
		c += (t - 0.5) / float(levels);
	}
	c = clamp(round(c * float(levels - 1)) / float(levels - 1), 0.0, 1.0);
	COLOR = vec4(c, 1.0);
}
