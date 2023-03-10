import math

from typing import Literal, Type, Callable
import galois
import numpy as np
import matplotlib.pyplot as plt


class QR_code:
    def __init__(self, level: Literal['L', 'M', 'Q', 'H'], mask: Literal['optimal'] | list[int]) -> None:
        self.level:     Literal['L', 'M', 'Q', 'H'] = level     # error correction level, can be 'L', 'M', 'Q', 'H'
        self.mask:      Literal["optimal"] | list[int] = mask   # the mask pattern, can be 'optimal' or either the three bits representing the mask in a list
        self.version:   int = 6                                 # the version number, range from 1 to 40, only version number 6 is implemented

        # the generator polynomial of the Reed-Solomon code
        if level == 'L':
            self.generator: galois.Poly = self.makeGenerator(2, 8, 9)
        elif level == 'M':
            self.generator: galois.Poly = self.makeGenerator(2, 8, 8)
        elif level == 'Q':
            self.generator: galois.Poly = self.makeGenerator(2, 8, 12)
        elif level == 'H':
            self.generator: galois.Poly = self.makeGenerator(2, 8, 14)
        else:
            raise Exception("Invalid error correction level!")

        self.NUM_MASKS: int = 8  # the number of masks

    def encodeData(self, bitstream: np.ndarray) -> np.ndarray:
        # first add padding to the bitstream obtained from generate_dataStream()
        # then split this datasequence in blocks and perform RS-coding
        # and apply interleaving on the bytes (see the specification
        # section 8.6 'Constructing the final message codeword sequence')
        # INPUT:
        #  -bitstream: bitstream to be encoded in QR code. 1D numpy array e.g. bitstream=np.array([1,0,0,1,0,1,0,0,...])
        # OUTPUT:
        #  -data_enc: encoded bits after interleaving. Length should be 172*8 (version 6). 1D numpy array e.g. data_enc=np.array([1,0,0,1,0,1,0,0,...])
        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, "bitstream must be a 1D numpy array"

        ################################################################################################################
        # insert your code here
        data_enc: np.ndarray = ...
        ################################################################################################################

        assert len(np.shape(data_enc)) == 1 and type(data_enc) is np.ndarray, "data_enc must be a 1D numpy array"
        return data_enc

    def decodeData(self, data_enc: np.ndarray) -> np.ndarray:
        # Function to decode data, this is the inverse function of encodeData
        # INPUT:
        #  -data_enc: encoded binary data with the bytes being interleaved. 1D numpy array e.g. data_enc=np.array([1,0,0,1,0,1,0,0,...])
        #   length is equal to 172*8
        # OUTPUT:
        #  -bitstream: a bitstream with the padding removed. 1D numpy array e.g. bitstream=np.array([1,0,0,1,0,1,0,0,...])
        assert len(np.shape(data_enc)) == 1 and type(data_enc) is np.ndarray, "data_enc must be a 1D numpy array"

        ################################################################################################################
        # insert your code here
        bitstream: np.ndarray = ...
        ################################################################################################################

        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, "bitstream must be a 1D numpy array"
        return bitstream

    # QR-code generator/reader (do not change)
    def generate(self, data: str) -> np.ndarray:
        # This function creates and displays a QR code matrix with either the optimal mask or a specific mask (depending on self.mask)
        # INPUT:
        #  -data: data to be encoded in the QR code. In this project a string with only characters from the alphanumeric mode
        #  e.g data="A1 %"
        # OUTPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        assert type(data) is str, "data must be a string"

        bitstream: np.ndarray = self.generate_dataStream(data)
        data_bits: np.ndarray = self.encodeData(bitstream)

        if self.mask == "optimal":
            # obtain optimal mask if mask=='optimal', otherwise use selected mask
            mask_code: list[list[int]] = [[int(x) for x in np.binary_repr(i, 3)] for i in range(self.NUM_MASKS)]
            score: np.ndarray = np.ones(self.NUM_MASKS)
            score[:] = float("inf")
            for m in range(self.NUM_MASKS):
                QRmatrix_m: np.ndarray = self.construct(data_bits, mask_code[m], show=False)
                score[m] = self.evaluateMask(QRmatrix_m)
                if score[m] == np.min(score):
                    QRmatrix = QRmatrix_m.copy()
                    self.mask = mask_code[m]

        # create the QR-code using either the selected or the optimal mask
        QRmatrix: np.ndarray = self.construct(data_bits, self.mask)

        return QRmatrix

    def construct(self, data: np.ndarray, mask: tuple[int, int, int], show: bool = True) -> np.ndarray:
        # This function creates a QR code matrix with specified data and
        # mask (this might not be the optimal mask though)
        # INPUT:
        #  -data: the output from encodeData, i.e. encoded bits after interleaving. Length should be 172*8 (version 6).
        #  1D numpy array e.g. data=np.array([1,0,0,1,0,1,0,0,...])
        #  -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        # OUTPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        L:        int = 17+4*self.version
        QRmatrix: np.ndarray = np.zeros((L, L), dtype=int)

        PosPattern: np.ndarray = np.ones((7, 7), dtype=int)
        PosPattern[[1, 5], 1:6] = 0
        PosPattern[1:6, [1, 5]] = 0

        QRmatrix[0:7, 0:7] = PosPattern
        QRmatrix[-7:, 0:7] = PosPattern
        QRmatrix[0:7, -7:] = PosPattern

        AlignPattern: np.ndarray = np.ones((5, 5), dtype=int)
        AlignPattern[[1, 3], 1:4] = 0
        AlignPattern[1:4, [1, 3]] = 0

        QRmatrix[32:37, L-9:L-4] = AlignPattern

        L_timing:      int = L-2*8
        TimingPattern: np.ndarray = np.zeros((1, L_timing), dtype=int)
        TimingPattern[0, 0::2] = np.ones((1, (L_timing+1)//2), dtype=int)

        QRmatrix[6, 8:(L_timing+8)] = TimingPattern
        QRmatrix[8:(L_timing+8), 6] = TimingPattern

        FI: np.ndarray = self.encodeFormat(self.level, mask)
        FI = np.flip(FI)

        QRmatrix[0:6, 8] = FI[0:6]
        QRmatrix[7:9, 8] = FI[6:8]
        QRmatrix[8, 7] = FI[8]
        QRmatrix[8, 5::-1] = FI[9:]
        QRmatrix[8, L-1:L-9:-1] = FI[0:8]
        QRmatrix[L-7:L, 8] = FI[8:]
        QRmatrix[L-8, 8] = 1

        nogo: np.ndarray = np.zeros((L, L), dtype=int)
        nogo[0:9, 0:9] = np.ones((9, 9), dtype=int)
        nogo[L-1:L-9:-1, 0:9] = np.ones((8, 9), dtype=int)
        nogo[0:9, L-1:L-9:-1] = np.ones((9, 8), dtype=int)
        nogo[6, 8:(L_timing+8)] = np.ones((L_timing), dtype=int)
        nogo[8:(L_timing+8), 6] = np.ones((1, L_timing), dtype=int)
        nogo[32:37, L-9:L-4] = np.ones((5, 5), dtype=int)
        nogo = np.delete(nogo, 6, 1)
        nogo = nogo[-1::-1, -1::-1]
        col1: np.ndarray = nogo[:, 0::2].copy()
        col2 = nogo[:, 1::2].copy()
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]
        nogo_reshape: np.ndarray = np.array([col1.flatten(order='F'), col2.flatten(order='F')])
        QR_reshape = np.zeros((2, np.shape(nogo_reshape)[1]), dtype=int)

        ind_col:    int = 0
        ind_row:    int = 0
        ind_data:   int = 0

        for i in range(QR_reshape.size):
            if (nogo_reshape[ind_row, ind_col] == 0):
                QR_reshape[ind_row, ind_col] = data[ind_data]
                ind_data = ind_data + 1
                nogo_reshape[ind_row, ind_col] = 1

            ind_row = ind_row+1
            if ind_row > 1:
                ind_row = 0
                ind_col = ind_col + 1

            if ind_data >= len(data):
                break

        QR_data: np.ndarray = np.zeros((L-1, L), dtype=int)
        colr:    np.ndarray = np.reshape(QR_reshape[0, :], (L, len(QR_reshape[0, :])//L), order='F')
        colr[:, 1::2] = colr[-1::-1, 1::2]
        QR_data[0::2, :] = np.transpose(colr)

        coll: np.ndarray = np.reshape(QR_reshape[1, :], (L, len(QR_reshape[1, :])//L), order='F')
        coll[:, 1::2] = coll[-1::-1, 1::2]
        QR_data[1::2, :] = np.transpose(coll)

        QR_data = np.transpose(QR_data[-1::-1, -1::-1])
        QR_data = np.hstack((QR_data[:, 0:6], np.zeros((L, 1), dtype=int), QR_data[:, 6:]))

        QRmatrix = QRmatrix + QR_data

        QRmatrix[30:33, 0:2] = np.ones((3, 2), dtype=int)
        QRmatrix[29, 0] = 1

        nogo = nogo[-1::-1, -1::-1]
        nogo = np.hstack((nogo[:, 0:6], np.ones((L, 1), dtype=int), nogo[:, 6:]))

        QRmatrix = self.applyMask(mask, QRmatrix, nogo)

        if show == True:
            plt.matshow(QRmatrix, cmap='Greys')
            plt.show()

        return QRmatrix

    @staticmethod
    def read(QRmatrix: np.ndarray) -> str:
        # function to read the encoded data from a QR code
        # INPUT:
        #  -QRmatrix: a 41 by 41 numpy array with 0's and 1's
        # OUTPUT:
        # -data_dec: data to be encoded in the QR code. In this project a string with only characters from the alphanumeric mode
        #  e.g data="A1 %"
        assert np.shape(QRmatrix) == (41, 41) and type(QRmatrix) is np.ndarray, 'QRmatrix must be a 41 by numpy array'

        FI: np.ndarray = np.zeros((15), dtype=int)
        FI[0:6] = QRmatrix[0:6, 8]
        FI[6:8] = QRmatrix[7:9, 8]
        FI[8] = QRmatrix[8, 7]
        FI[9:] = QRmatrix[8, 5::-1]
        FI = FI[-1::-1]

        L: int = np.shape(QRmatrix)[0]
        L_timing: int = L - 2*8

        success, level, mask = QR_code.decodeFormat(FI)

        if success:
            qr: QR_code = QR_code(level, mask)
        else:
            FI = np.zeros((15), dtype=int)
            FI[0:8] = QRmatrix[8, L-1:L-9:-1]
            FI[8:] = QRmatrix[L-7:L, 8]

            [success, level, mask] = QR_code.decodeFormat(FI)
            if success:
                qr: QR_code = QR_code(level, mask)
            else:
                # print('Format information was not decoded succesfully')
                exit(-1)

        nogo: np.ndarray = np.zeros((L, L))
        nogo[0:9, 0:9] = np.ones((9, 9), dtype=int)
        nogo[L-1:L-9:-1, 0:9] = np.ones((8, 9), dtype=int)
        nogo[0:9, L-1:L-9:-1] = np.ones((9, 8), dtype=int)

        nogo[6, 8:(L_timing+8)] = np.ones((1, L_timing), dtype=int)
        nogo[8:(L_timing+8), 6] = np.ones((L_timing), dtype=int)

        nogo[32:37, L-9:L-4] = np.ones((5, 5), dtype=int)

        QRmatrix: np.ndarray = QR_code.applyMask(mask, QRmatrix, nogo)

        nogo = np.delete(nogo, 6, 1)
        nogo = nogo[-1::-1, -1::-1]
        col1: np.ndarray = nogo[:, 0::2]
        col2: np.ndarray = nogo[:, 1::2]
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]

        nogo_reshape: np.ndarray = np.vstack((np.transpose(col1.flatten(order='F')), np.transpose(col2.flatten(order='F'))))

        QRmatrix = np.delete(QRmatrix, 6, 1)
        QRmatrix = QRmatrix[-1::-1, -1::-1]
        col1 = QRmatrix[:, 0::2]
        col2 = QRmatrix[:, 1::2]
        col1[:, 1::2] = col1[-1::-1, 1::2]
        col2[:, 1::2] = col2[-1::-1, 1::2]

        QR_reshape: np.ndarray = np.vstack((np.transpose(col1.flatten(order='F')), np.transpose(col2.flatten(order='F'))))

        data:       np.ndarray = np.zeros((172*8, 1))
        ind_col:    int = 0
        ind_row:    int = 0
        ind_data:   int = 0
        for i in range(QR_reshape.size):
            if (nogo_reshape[ind_row, ind_col] == 0):
                data[ind_data] = QR_reshape[ind_row, ind_col]
                ind_data = ind_data + 1
                nogo_reshape[ind_row, ind_col] = 1

            ind_row = ind_row+1
            if ind_row > 1:
                ind_row = 0
                ind_col = ind_col + 1

            if ind_data >= len(data):
                break

        bitstream = qr.decodeData(data.flatten())
        data_dec = QR_code.read_dataStream(bitstream)

        assert type(data_dec) is str, 'data_dec must be a string'
        return data_dec

    @staticmethod
    def generate_dataStream(data: str) -> np.ndarray:
        # this function creates a bitstream from the user data.
        # ONLY IMPLEMENT ALPHANUMERIC MODE !!!!!!
        # INPUT:
        #  -data: the data string (for example 'ABC012')
        # OUTPUT:
        #  -bitstream: a 1D numpy array containing the bits that
        #  represent the input data, headers should be added, no padding must be added here.
        #  Add padding in EncodeData. e.g. bitstream=np.array([1,0,1,1,0,1,0,0,...])
        assert type(data) is str, "data must be a string"

        ################################################################################################################
        # insert your code here
        
        #initialize bitstream as an array containing 0010. This part of the header indicates alphanumeric mode
        bitstream = np.array([0,0,1,0])
        
        #convert the input data to a list
        input = list(data)
        
        #convert each input character to the corresponding number specified for alphanumeric mode
        convert_array = np.array(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ','$','%','*','+','-','.','/',':'])
        numbers = []
        for i in input:
            num = np.where(convert_array == i)[0][0]
            numbers.append(num)
           
        #group the numbers in groups of two and construct number to be encoded
        #convert each of the grouped numbers to a bit sequence of 11 bits.
        #if the number of characters is odd, the last number gets represented with 6 bits
        #first make i an array of strings containing the bits
        grouped_numbers = []
        p = 0
        while p< len(numbers):
            if(p<(len(numbers)-1)):
                grouped_number = numbers[p]*45+numbers[p+1]
                #convert to binary number of 11 bits
                grouped_numbers.append('{0:011b}'.format(grouped_number))
            else:
                grouped_number = numbers[p]
                #convert to binary number of 6 bits
                grouped_numbers.append('{0:06b}'.format(grouped_number))
            p+=2
        data = []
        for i in grouped_numbers:
            for n in i:
                data.append(int(n))
        
        #the number of characters needs to be converted to a 9 bit binary number
        character_count_indicator = '{0:09b}'.format(len(input))
        character_count = []
        for i in character_count_indicator:
            character_count.append(int(i))
            
        #add the character count indicator part of the header
        bitstream = np.append(bitstream,np.array(character_count))
        
        #finally add the binary version of the grouped numbers to the bitstream
        bitstream = np.append(bitstream,np.array(data))
        ################################################################################################################

        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, "bitstream must be a 1D numpy array"
        return bitstream

    @staticmethod
    def read_dataStream(bitstream: np.ndarray) -> str:
        # inverse function of generate_dataStream: convert a bitstream to an alphanumeric string
        # INPUT:
        #  -bitstream: a 1D numpy array of bits (including the header bits) e.g. bitstream=np.array([1,0,1,1,0,1,0,0,...])
        # OUTPUT:
        #  -data: the encoded data string (for example 'ABC012')
        assert len(np.shape(bitstream)) == 1 and type(bitstream) is np.ndarray, "bitstream must be a 1D numpy array"

        ################################################################################################################
        # insert your code here
        data: str = ...
        ################################################################################################################

        assert type(data) is str, "data must be a string"
        return data

    @staticmethod
    def encodeFormat(level: Literal['L', 'M', 'Q', 'H'], mask: tuple[int, int, int]) -> np.ndarray:
        # Encodes the 5 bit format to a 15 bit sequence using a BCH code
        # INPUT:
        #  -level: specified level 'L','M','Q' or'H'
        #  -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        # OUTPUT:
        # format: 1D numpy array with the FI-codeword, with the special FI-mask applied (see specification)
        assert type(mask) is list and len(mask) == 3, "mask must be a list of length 3"

        ################################################################################################################
        # insert your code here
        format: np.ndarray = ...
        ################################################################################################################

        assert len(np.shape(format)) == 1 and type(format) is np.ndarray and format.size == 15, "format must be a 1D numpy array of length 15"
        return format

    @staticmethod
    def decodeFormat(Format: np.ndarray) -> tuple[bool, Literal['L', 'M', 'Q', 'H'], tuple[int, int, int]]:
        # Decode the format information
        # INPUT:
        # -format: 1D numpy array (15bits) with format information (with FI-mask applied)
        # OUTPUT:
        # -success: True if decodation succeeded, False if decodation failed
        # -level: being an element of {'L','M','Q','H'}
        # -mask: three bits that represent the mask. 1D list e.g. mask=[1,0,0]
        assert len(np.shape(Format)) == 1 and type(Format) is np.ndarray and Format.size == 15, "format must be a 1D numpy array of length 15"

        ################################################################################################################
        # insert your code here
        success: bool = ...
        level:   Literal['L', 'M', 'Q', 'H'] = ...
        mask:    tuple[int, int, int] = ...
        ################################################################################################################

        assert type(mask) is list and len(mask) == 3, "mask must be a list of length 3"
        return success, level, mask

    @staticmethod
    def makeGenerator(p: int, m: int, t: int) -> galois.Poly:
        # Generate the Reed-Solomon generator polynomial with error correcting capability t over GF(p^m)
        # INPUT:
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -t: error correction capability of the Reed-Solomon code, positive integer > 1
        # OUTPUT:
        #  -generator: galois.Poly object representing the generator polynomial

        ################################################################################################################
        # insert your code here
        generator: galois.Poly = ...
        ################################################################################################################

        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), "generator must be a galois.Poly object"
        return generator

    @staticmethod
    def encodeRS(informationword: galois.FieldArray, p: int, m: int, n: int, k: int, generator: galois.Poly) -> galois.FieldArray:
        # Encode the informationword
        # INPUT:
        #  -informationword: a 1D array of galois.GF elements that represents the information word coefficients in GF(p^m) (first element is the highest degree coefficient)
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -n: codeword length, <= p^m-1
        #  -k: information word length
        #  -generator: galois.Poly object representing the generator polynomial
        # OUTPUT:
        #  -codeword: a 1D array of galois.GF elements that represents the codeword coefficients in GF(p^m) corresponding to systematic Reed-Solomon coding of the corresponding information word (first element is the highest degree coefficient)
        prim_poly: galois.Poly = galois.primitive_poly(p, m)
        GF:        Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
        assert type(informationword) is GF and len(np.shape(informationword)) == 1, "each element of informationword(1D)  must be a galois.GF element"
        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), "generator must be a galois.Poly object"

        ################################################################################################################
        # insert your code here
        codeword: galois.FieldArray = ...
        ################################################################################################################

        assert type(codeword) is GF and len(np.shape(codeword)) == 1, "each element of codeword(1D)  must be a galois.GF element"
        return codeword

    @staticmethod
    def decodeRS(codeword: galois.FieldArray, p: int, m: int, n: int, k: int, generator: galois.Poly) -> galois.FieldArray:
        # Decode the codeword
        # INPUT:
        #  -codeword: a 1D array of galois.GF elements that represents the codeword coefficients in GF(p^m) corresponding to systematic Reed-Solomon coding of the corresponding information word (first element is the highest degree coefficient)
        #  -p: field characteristic, prime number
        #  -m: positive integer
        #  -n: codeword length, <= p^m-1
        #  -k: decoded word length
        #  -generator: galois.Poly object representing the generator polynomial
        # OUTPUT:
        #  -decoded: a 1D array of galois.GF elements that represents the decoded information word coefficients in GF(p^m) (first element is the highest degree coefficient)
        prim_poly: galois.Poly = galois.primitive_poly(p, m)
        GF:        Type[galois.FieldArray] = galois.GF(p**m, irreducible_poly=prim_poly)
        assert type(codeword) is GF and len(np.shape(codeword)) == 1, "each element of codeword(1D)  must be a galois.GF element"
        assert type(generator) == type(galois.Poly([0], field=galois.GF(m))), "generator must be a galois.Poly object"

        ################################################################################################################
        # insert your code here
        decoded: galois.FieldArray = ...
        ################################################################################################################

        assert type(decoded) is GF and len(np.shape(decoded)) == 1, "each element of decoded(1D)  must be a galois.GF element"
        return decoded

    # function to mask or unmask a QR_code matrix and to evaluate the masked QR symbol (do not change)
    @staticmethod
    def applyMask(mask: tuple[int, int, int], QRmatrix: np.ndarray, nogo: np.ndarray) -> np.ndarray:
        # define all the masking functions
        maskfun1: Callable[[int, int], bool] = lambda i, j: (i+j) % 2 == 0
        maskfun2: Callable[[int, int], bool] = lambda i, j: (i) % 2 == 0
        maskfun3: Callable[[int, int], bool] = lambda i, j: (j) % 3 == 0
        maskfun4: Callable[[int, int], bool] = lambda i, j: (i+j) % 3 == 0
        maskfun5: Callable[[int, int], bool] = lambda i, j: (math.floor(i/2)+math.floor(j/3)) % 2 == 0
        maskfun6: Callable[[int, int], bool] = lambda i, j: (i*j) % 2 + (i*j) % 3 == 0
        maskfun7: Callable[[int, int], bool] = lambda i, j: ((i*j) % 2 + (i*j) % 3) % 2 == 0
        maskfun8: Callable[[int, int], bool] = lambda i, j: ((i+j) % 2 + (i*j) % 3) % 2 == 0

        maskfun: list[Callable[[int, int], bool]] = [maskfun1, maskfun2, maskfun3, maskfun4, maskfun5, maskfun6, maskfun7, maskfun8]

        L: int = len(QRmatrix)
        QRmatrix_masked: np.ndarray = QRmatrix.copy()

        mask_number: int = int(''.join(str(el) for el in mask), 2)

        maskfunction: Callable[[int, int], bool] = maskfun[mask_number]

        for i in range(L):
            for j in range(L):
                if nogo[i, j] == 0:
                    QRmatrix_masked[i, j] = (QRmatrix[i, j] + maskfunction(i, j)) % 2

        return QRmatrix_masked

    @staticmethod
    def evaluateMask(QRmatrix: np.ndarray) -> float:
        Ni:             list[int] = [3, 3, 40, 10]
        L:              int = len(QRmatrix)
        score:          float = 0
        QRmatrix_temp:  np.ndarray = np.vstack((QRmatrix, 2*np.ones((1, L)), np.transpose(QRmatrix), 2*np.ones((1, L))))

        vector: np.ndarray = QRmatrix_temp.flatten(order='F')
        splt:   list[np.ndarray] = QR_code.SplitVec(vector)

        neighbours: np.ndarray = np.array([len(x) for x in splt])
        temp: np.ndarray = neighbours > 5
        if (temp).any():
            score += sum([x-5+Ni[0] for x in neighbours if x > 5])

        QRmatrix_tmp: np.ndarray = QRmatrix
        rec_sizes: np.ndarray = np.array([[5, 2, 4, 4, 3, 4, 2, 3, 2, 3, 2], [2, 5, 4, 3, 4, 2, 4, 3, 3, 2, 2]])

        for i in range(np.shape(rec_sizes)[1]):

            QRmatrix_tmp, num = QR_code.find_rect(QRmatrix_tmp, rec_sizes[0, i], rec_sizes[1, i])
            score += num*(rec_sizes[0, i]-1)*(rec_sizes[1, i]-1)*Ni[1]

        QRmatrix_tmp = np.vstack((QRmatrix, 2*np.ones((1, L)), np.transpose(QRmatrix), 2*np.ones((1, L))))
        temp = QRmatrix_tmp.flatten(order='F')
        temp2 = [x for x in range(len(temp)-6) if (temp[x:x+7] == [1, 0, 1, 1, 1, 0, 1]).all()]
        score += Ni[2]*len(temp2)

        nDark: float = sum(sum(QRmatrix == 1))/L**2
        k:     float = math.floor(abs(nDark-0.5)/0.05)
        score += Ni[3]*k

        return score

    @staticmethod
    def SplitVec(vector: np.ndarray) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        temp: np.ndarray = np.where(np.diff(vector) != 0)[0]
        temp = temp+1
        temp = np.insert(temp, 0, 0)

        for i in range(len(temp)):
            if i == len(temp)-1:
                output.append(vector[temp[i]:])
            else:
                output.append(vector[temp[i]:temp[i+1]])

        return output

    @staticmethod
    def find_rect(A: np.ndarray, nR: int, nC: int) -> tuple[np.ndarray, int]:

        Lx: int = np.shape(A)[0]
        Ly: int = np.shape(A)[1]
        num: int = 0
        A_new: np.ndarray = A.copy()

        for x in range(Lx-nR+1):
            for y in range(Ly-nC+1):
                test: np.ndarray = np.unique(A_new[x:x+nR, y:y+nC])

                if len(test) == 1:
                    num += 1
                    A_new[x:x+nR, y:y+nC] = np.reshape(np.arange(2+x*nR+y, 2+nR*nC+x*nR+y), (nR, nC))

        return A_new, num
